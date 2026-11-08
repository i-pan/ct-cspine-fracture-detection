from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from skp.configs import Config
from skp.cspine.crops import (
    map_bounds_to_shape,
    resize_depth_and_fit_axial,
    resize_volume,
    vertebra_bounds,
)
from skp.cspine.dicom import dicom_to_hu, hu_to_uint8
from skp.cspine.evaluation import topk_slice_recall, weighted_binary_log_loss
from skp.cspine.folds import assign_multilabel_folds
from skp.cspine.localization_evaluation import (
    _compute_spatial_metrics,
    _spatial_accumulator,
    _update_spatial_metrics,
)
from skp.cspine.pseudolabels import (
    build_slice_teacher_manifest,
    contiguous_pseudo_labels,
)
from skp.cspine.segmentation_inference import (
    keep_largest_components,
    topology_anomaly_levels,
)
from skp.cspine.slice_localization import (
    aggregate_native_predictions,
    ensemble_native_predictions,
    evaluate_native_predictions,
    interpolate_crop_logits,
    resampled_source_instances,
)
from skp.cspine.thick import reconstruct_thick_volume, slab_weights
from skp.cspine.totalsegmentator_teacher import (
    build_totalsegmentator_qc_manifest,
    dice_by_level,
    remap_totalsegmentator_mask,
)
from skp.datasets.cspine_vertebra_mil import (
    Dataset,
    resample_close_square_mil_crop,
    resample_native_crop,
    resample_tight_original_x3d_crop,
    variable_depth_collate,
)
from skp.datasets.cspine_segmentation_3d import Dataset as SegmentationDataset
from skp.losses.cspine import DiceCrossEntropyLoss, VertebraMILLoss
from skp.models.classification.cspine_mil import Net
from skp.models.classification.cspine_x3d_mil import Net as X3DMILNet


def test_slice_logits_map_to_native_grid_and_average_duplicate_sources():
    native = pd.DataFrame(
        {"instance_number": [10, 11, 12], "slice_index": [0, 1, 2]}
    )
    mapped = interpolate_crop_logits([0.0, 2.0, 4.0, 6.0], [10, 11, 11, 12], native)
    assert mapped["logit"].tolist() == pytest.approx([0.0, 3.0, 6.0])


def test_slice_localization_aggregates_levels_and_scores_boxes():
    slices = pd.DataFrame(
        {
            "study_uid": ["s1"] * 4,
            "instance_number": [10, 11, 12, 13],
            "slice_index": [0, 1, 2, 3],
        }
    )
    crops = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "s1",
                "level": "C1",
                "source_instances": [10, 11, 12],
                "slice_logits": [-2.0, 3.0, -1.0],
            },
            {
                "StudyInstanceUID": "s1",
                "level": "C2",
                "source_instances": [11, 12, 13],
                "slice_logits": [1.0, 0.0, -2.0],
            },
        ]
    )
    native = aggregate_native_predictions(crops, slices)
    assert native["logit_max"].tolist() == [-2.0, 3.0, 0.0, -2.0]
    boxes = pd.DataFrame({"StudyInstanceUID": ["s1"], "slice_number": [11]})
    metrics = evaluate_native_predictions(native, boxes)
    assert metrics["auc_max"] == 1.0
    assert metrics["covered_annotated_slices"] == 1


def test_close_square_depth_positions_map_to_native_instances():
    mask = np.zeros((8, 2, 2), dtype=np.float32)
    mask[1:7] = 1
    instances = list(range(10, 18))
    assert resampled_source_instances(
        mask, instances, target_depth=3
    ) == [12, 14, 16]
    assert resampled_source_instances(
        mask,
        instances,
        target_depth=9,
        stack_size=3,
    ) == [11, 13, 15]


def test_slice_localization_ensemble_averages_before_level_pooling():
    slices = pd.DataFrame(
        {
            "study_uid": ["s1", "s1"],
            "instance_number": [10, 11],
            "slice_index": [0, 1],
        }
    )
    first = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "s1",
                "level": "C1",
                "source_instances": [10, 11],
                "slice_logits": [4.0, 0.0],
            },
            {
                "StudyInstanceUID": "s1",
                "level": "C2",
                "source_instances": [10, 11],
                "slice_logits": [0.0, 2.0],
            },
        ]
    )
    second = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "s1",
                "level": "C1",
                "source_instances": [10, 11],
                "slice_logits": [0.0, 4.0],
            },
            {
                "StudyInstanceUID": "s1",
                "level": "C2",
                "source_instances": [10, 11],
                "slice_logits": [2.0, 0.0],
            },
        ]
    )
    ensemble = ensemble_native_predictions(first, second, slices)
    assert ensemble["logit_max"].tolist() == [2.0, 2.0]


def test_x3d_mil_preserves_depth_and_emits_one_logit_per_position():
    cfg = Config(
        backbone="x3d_s",
        pretrained=False,
        include_last_block=True,
        num_input_channels=2,
        num_classes=1,
        dim0_strides=[1, 1, 1, 1, 1],
        output_stride=32,
        enable_gradient_checkpointing=False,
        x3d_feature_dim=432,
        dropout=0.0,
        mil_pool_temperature=0.5,
        mil_pool_normalize=True,
        normalization="0_1",
        normalization_params={"min": 0, "max": 255},
        load_pretrained_model=None,
    )
    model = X3DMILNet(cfg).eval()
    x = torch.rand(1, 2, 8, 64, 64) * 255
    with torch.no_grad():
        output = model({"x": x}, return_features=True)
    assert output["slice_logits"].shape == (1, 8, 1)
    assert output["logits"].shape == (1, 1)
    assert output["feature_map"].shape[2] == 8
    assert all(
        module.stride[0] == 1
        for module in model.encoder.modules()
        if isinstance(module, torch.nn.Conv3d)
    )


def test_x3d_global_pool_collapses_depth_height_and_width():
    cfg = Config(
        backbone="x3d_s",
        pretrained=False,
        include_last_block=True,
        num_input_channels=2,
        num_classes=1,
        dim0_strides=[1, 1, 1, 1, 1],
        output_stride=32,
        enable_gradient_checkpointing=False,
        x3d_feature_dim=432,
        dropout=0.0,
        global_pool_3d=True,
        multisample_dropout=5,
        normalization="0_1",
        normalization_params={"min": 0, "max": 255},
        load_pretrained_model=None,
    )
    model = X3DMILNet(cfg).eval()
    dropout_calls = []
    hook = model.dropout.register_forward_hook(
        lambda _module, _inputs, _output: dropout_calls.append(1)
    )
    with torch.no_grad():
        output = model(
            {"x": torch.rand(1, 2, 8, 64, 64) * 255},
            return_features=True,
        )
    hook.remove()
    assert output["logits"].shape == (1, 1)
    assert output["features"].shape == (1, 432)
    assert output["feature_map"].shape[2] == 8
    assert "slice_logits" not in output
    assert len(dropout_calls) == 5


def test_original_x3d_crop_uses_tight_bounds_and_nearest_depth_resampling():
    image = np.broadcast_to(
        np.arange(5, dtype=np.float32)[:, None, None] * 10,
        (5, 6, 8),
    ).copy()
    mask = np.zeros_like(image)
    mask[1:4, 1:5, 2:6] = 0.5

    resized = resample_tight_original_x3d_crop(
        image,
        mask,
        output_shape=(7, 10, 12),
        mask_threshold=0.4,
    )

    assert resized.shape == (7, 10, 12)
    # scipy order-0 index zoom maps the three tight source planes to 0, 0,
    # 1, 1, 1, 2, 2 before their original depth offset is restored.
    expected_depth_values = np.array([10, 10, 20, 20, 20, 30, 30])
    assert np.allclose(resized[:, 0, 0], expected_depth_values)
    assert np.allclose(resized, expected_depth_values[:, None, None])


def test_close_square_crop_uses_linear_depth_downsampling_without_distortion():
    image = np.broadcast_to(
        np.arange(5, dtype=np.float32)[:, None, None] * 10,
        (5, 8, 8),
    ).copy()
    mask = np.zeros_like(image)
    mask[:, 2:6, 3:5] = 1

    resized_image, resized_mask = resample_close_square_mil_crop(
        image,
        mask,
        output_shape=(3, 6, 6),
        padding_fraction=0.0,
    )

    assert resized_image.shape == resized_mask.shape == (3, 6, 6)
    assert np.allclose(resized_image[:, 0, 0], [10 / 3, 20, 110 / 3])
    assert np.allclose(resized_image, resized_image[:, :1, :1])
    assert resized_mask[:, :, 1:5].mean() > resized_mask[:, :, [0, 5]].mean()


def test_close_square_crop_uses_nearest_depth_repetition_when_short():
    image = np.broadcast_to(
        np.array([10, 20], dtype=np.float32)[:, None, None],
        (2, 4, 4),
    ).copy()
    mask = np.ones_like(image)

    resized_image, _ = resample_close_square_mil_crop(
        image,
        mask,
        output_shape=(5, 4, 4),
        padding_fraction=0.0,
    )

    assert resized_image[:, 0, 0].tolist() == [10, 10, 10, 20, 20]


class _Dicom:
    pixel_array = np.array([[0, 100]], dtype=np.int16)
    RescaleSlope = 10
    RescaleIntercept = -10240


def test_windowing_uses_fixed_wl400_ww1800():
    values = np.array([-1000, -500, 400, 1300, 2000], dtype=np.float32)
    assert hu_to_uint8(values).tolist() == [0, 0, 128, 255, 255]
    assert dicom_to_hu(_Dicom()).tolist() == [[-10240.0, -9240.0]]


def test_multilabel_folds_are_study_unique_and_deterministic():
    rows = []
    for index in range(50):
        rows.append(
            {
                "StudyInstanceUID": f"study-{index}",
                "patient_overall": index % 2,
                **{f"C{level}": int(index % (level + 2) == 0) for level in range(1, 8)},
                "has_boxes": int(index % 5 == 0),
                "has_segmentation": int(index % 7 == 0),
            }
        )
    studies = pd.DataFrame(rows)
    columns = (
        "patient_overall",
        *(f"C{level}" for level in range(1, 8)),
        "has_boxes",
        "has_segmentation",
    )
    first = assign_multilabel_folds(studies, columns=columns, seed=9)
    second = assign_multilabel_folds(studies, columns=columns, seed=9)
    assert first["fold"].tolist() == second["fold"].tolist()
    assert set(first["fold"]) == set(range(5))
    assert first["StudyInstanceUID"].is_unique


def test_crop_bounds_and_resize():
    mask = np.zeros((10, 20, 30), dtype=np.uint8)
    mask[2:8, 5:15, 10:20] = 3
    bounds = vertebra_bounds(mask, 3, padding_fraction=0.1)
    assert bounds == (slice(0, 10), slice(3, 17), slice(8, 22))
    resized = resize_volume(mask, (5, 8, 9), mode="nearest")
    assert resized.shape == (5, 8, 9)
    assert set(np.unique(resized)).issubset({0, 3})


def test_resize_depth_and_fit_axial_does_not_magnify_native_pixels():
    volume = np.ones((4, 6, 8), dtype=np.float32)
    resized, scale = resize_depth_and_fit_axial(
        volume,
        (2, 12, 12),
        mode="trilinear",
    )
    assert resized.shape == (2, 12, 12)
    assert scale == 1.0
    assert np.all(resized[:, 3:9, 2:10] == 1)
    assert resized.sum() == 2 * 6 * 8


def test_resize_depth_and_fit_axial_downsamples_oversized_crop():
    volume = np.ones((4, 20, 10), dtype=np.float32)
    resized, scale = resize_depth_and_fit_axial(
        volume,
        (2, 10, 10),
        mode="trilinear",
    )
    assert resized.shape == (2, 10, 10)
    assert scale == 0.5
    assert resized.sum() == 2 * 10 * 5


def test_map_bounds_to_shape_preserves_coverage():
    bounds = (slice(2, 8), slice(10, 20), slice(20, 40))
    mapped = map_bounds_to_shape(bounds, (10, 50, 100), (25, 100, 300))
    assert mapped == (slice(5, 20), slice(20, 40), slice(60, 120))


def test_thick_reconstruction_averages_hu_before_windowing():
    positions = np.arange(4, dtype=float)
    volume = np.arange(4, dtype=np.float32)[:, None, None] * 100
    weights = slab_weights(positions, center=1.5, thickness=2.0)
    assert np.isclose(weights.sum(), 1)
    slabs, centers, contributors = reconstruct_thick_volume(
        volume, positions, thickness=2.0
    )
    assert slabs.shape[1:] == (1, 1)
    assert len(centers) == len(contributors) == len(slabs)


def test_segmentation_cleanup_keeps_largest_component_per_level():
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[2:4, 2:4, 2:4] = 1
    mask[1:3, 0:2, 0:2] = 2
    cleaned = keep_largest_components(mask)
    assert cleaned[0, 0, 0] == 0
    assert np.sum(cleaned == 1) == 8
    assert np.sum(cleaned == 2) == 8


def test_topology_qc_finds_missing_and_laterally_discontinuous_levels():
    mask = np.zeros((32, 64, 64), dtype=np.uint8)
    for level in range(1, 8):
        z = 30 - level * 4
        x = 50 if level == 5 else 32
        mask[z : z + 3, 30:34, x : x + 4] = level
    mask[mask == 3] = 0
    assert topology_anomaly_levels(mask) == [3, 5]


def test_pseudolabels_keep_ambiguous_slices_ignored():
    target, mask = contiguous_pseudo_labels(
        np.array([0.01, 0.2, 0.75, 0.9, 0.8, 0.3]), bag_target=1
    )
    assert target.tolist() == [0, 0, 1, 1, 1, 0]
    assert mask.tolist() == [True, False, True, True, True, False]


def test_slice_teacher_manifest_uses_native_adjacent_pngs_and_manual_boxes():
    slices = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "study_uid": ["study"] * 3,
            "instance_number": [3, 2, 1],
            "slice_index": [0, 1, 2],
            "fold": [2] * 3,
            "png_path": ["3.png", "2.png", "1.png"],
        }
    )
    predictions = pd.DataFrame(
        {
            "StudyInstanceUID": ["study"] * 3,
            "instance_number": [3, 2, 1],
            "slice_index": [0, 1, 2],
            "logit_max": [-2.0, 2.0, 0.0],
        }
    )
    boxes = pd.DataFrame(
        {"StudyInstanceUID": ["study"], "slice_number": [2]}
    )
    result = build_slice_teacher_manifest(predictions, slices, boxes)
    assert result["png_path_prev"].tolist() == ["3.png", "3.png", "2.png"]
    assert result["png_path_next"].tolist() == ["2.png", "1.png", "1.png"]
    assert result["manual_target"].tolist() == [0.0, 1.0, 0.0]
    assert result["manual_available"].all()
    assert result.loc[1, "teacher_probability"] > 0.5


def test_totalsegmentator_labels_are_remapped_to_c1_c7():
    source = np.array([[[50, 49, 48, 47, 46, 45, 44, 1]]], dtype=np.uint8)
    level_ids = {
        source_id: level
        for level, source_id in enumerate(range(50, 43, -1), start=1)
    }
    result = remap_totalsegmentator_mask(source, level_ids)
    assert result.tolist() == [[[1, 2, 3, 4, 5, 6, 7, 0]]]


def test_totalsegmentator_dice_is_per_level():
    target = np.array([[[1, 1, 2, 2, 3, 3, 4, 5, 6, 7]]], dtype=np.uint8)
    prediction = target.copy()
    prediction[0, 0, 0] = 0
    scores = dice_by_level(prediction, target)
    assert scores["dice_C1"] == pytest.approx(2 / 3)
    assert all(scores[f"dice_C{level}"] == 1.0 for level in range(2, 8))


def test_totalsegmentator_qc_rejects_only_all_background_samples():
    samples = pd.DataFrame(
        {
            "StudyInstanceUID": ["complete", "partial", "empty"],
            "missing_levels": [[], ["C7"], [f"C{i}" for i in range(1, 8)]],
            "volume_path": ["complete.npz", "partial.npz", "empty.npz"],
        }
    )
    result = build_totalsegmentator_qc_manifest(samples)
    assert result["StudyInstanceUID"].tolist() == ["complete", "partial"]


def test_totalsegmentator_qc_accepts_serialized_missing_levels():
    samples = pd.DataFrame(
        {
            "StudyInstanceUID": ["partial"],
            "missing_levels": ["['C7']"],
            "volume_path": ["partial.npz"],
        }
    )
    assert len(build_totalsegmentator_qc_manifest(samples)) == 1


def test_segmentation_pretraining_uses_manual_validation_file(tmp_path: Path):
    volume_path = tmp_path / "volume.npz"
    np.savez_compressed(
        volume_path,
        image=np.zeros((2, 2, 2), dtype=np.uint8),
        mask=np.zeros((2, 2, 2), dtype=np.uint8),
    )
    pseudo_path = tmp_path / "pseudo.parquet"
    manual_path = tmp_path / "manual.parquet"
    pd.DataFrame(
        [{"volume_path": str(volume_path), "fold": 1}]
    ).to_parquet(pseudo_path)
    pd.DataFrame(
        [{"volume_path": str(volume_path), "fold": 0}]
    ).to_parquet(manual_path)
    cfg = Config()
    cfg.annotations_file = str(pseudo_path)
    cfg.validation_annotations_file = str(manual_path)
    cfg.split_column = "fold"
    cfg.fold = 0
    train = SegmentationDataset(cfg, "train")
    validation = SegmentationDataset(cfg, "val")
    assert len(train) == len(validation) == 1
    assert train.df.iloc[0].fold == 1
    assert validation.df.iloc[0].fold == 0


def test_metrics_known_answers():
    targets = np.array([[1, 0], [0, 1]])
    predictions = np.array([[0.8, 0.2], [0.2, 0.8]])
    value = weighted_binary_log_loss(
        targets,
        predictions,
        positive_weights=np.ones(2),
        negative_weights=np.ones(2),
    )
    assert value == pytest.approx(-np.log(0.8))
    recall = topk_slice_recall(
        np.array([[0.1, 0.9, 0.2], [0.8, 0.2, 0.1]]),
        np.array([[0, 1, 0], [0, 0, 1]], dtype=bool),
        k=1,
    )
    assert recall == 0.5


def test_vertebra_dataset_contract(tmp_path: Path):
    crop_path = tmp_path / "crop.npz"
    np.savez_compressed(
        crop_path,
        image=np.zeros((4, 8, 8), dtype=np.uint8),
        mask=np.ones((4, 8, 8), dtype=np.float16),
    )
    annotations = pd.DataFrame(
        [
            {
                "crop_path": str(crop_path),
                "target": target,
                "fold": fold,
                "level_index": level,
                "source_depth": 10,
                "source_height": 20,
                "source_width": 30,
                "mean_z_spacing": 0.6,
            }
            for target, fold, level in [(0, 0, 0), (1, 1, 1)]
        ]
    )
    path = tmp_path / "annotations.parquet"
    annotations.to_parquet(path)
    cfg = Config(
        annotations_file=str(path),
        split_column="fold",
        fold=0,
        targets=["target"],
    )
    dataset = Dataset(cfg, "train")
    sample = dataset[0]
    assert sample["x"].shape == (4, 4, 8, 8)
    assert sample["y"].shape == (1,)
    assert sample["metadata"].shape == (4,)


def test_variable_depth_collate_right_pads_and_marks_valid_slices():
    samples = [
        {
            "x": torch.ones(4, depth, 8, 8),
            "y": torch.tensor([float(index)]),
            "level_index": torch.tensor(index),
            "metadata": torch.zeros(4),
            "index": torch.tensor(index),
        }
        for index, depth in enumerate((3, 5))
    ]
    batch = variable_depth_collate(samples)
    assert batch["x"].shape == (2, 4, 5, 8, 8)
    assert batch["mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert torch.count_nonzero(batch["x"][0, :, 3:]) == 0


def test_spaced_stack_uses_every_slice_once_for_matching_depth():
    image = np.arange(80, dtype=np.float32)[:, None, None]
    mask = image / 80
    stacked = Dataset._make_spaced_stack(
        image,
        mask,
        num_positions=16,
        stack_size=5,
    )
    assert stacked.shape == (6, 16, 1, 1)
    recovered = stacked[:5, :, 0, 0].T.reshape(-1)
    assert recovered.tolist() == list(range(80))
    assert stacked[5, :, 0, 0].tolist() == pytest.approx(
        (np.arange(2, 80, 5) / 80 * 255).tolist()
    )


def test_spaced_stack_samples_full_volume_when_depth_differs():
    image = np.arange(100, dtype=np.float32)[:, None, None]
    mask = np.ones_like(image)
    stacked = Dataset._make_spaced_stack(
        image,
        mask,
        num_positions=16,
        stack_size=5,
    )
    recovered = stacked[:5, :, 0, 0].T.reshape(-1)
    assert recovered[0] == 0
    assert recovered[-1] == 99
    assert np.all(np.diff(recovered) >= 1)


def test_d32_stack3_resamples_short_volume_to_96_with_nearest_repetition():
    image = np.arange(60, dtype=np.float32)[:, None, None]
    mask = np.ones_like(image)
    stacked = Dataset._make_spaced_stack(
        image,
        mask,
        num_positions=32,
        stack_size=3,
    )

    recovered = stacked[:3, :, 0, 0].T.reshape(-1)
    expected = np.linspace(0, 59, 96).round().astype(int)
    assert stacked.shape == (4, 32, 1, 1)
    assert recovered.tolist() == expected.tolist()
    assert np.count_nonzero(np.diff(recovered) == 0) > 0


def _model_cfg() -> Config:
    return Config(
        backbone="resnet18",
        pretrained=False,
        num_input_channels=4,
        image_height=32,
        image_width=32,
        use_level_embedding=False,
        level_embedding_dim=4,
        use_metadata=False,
        metadata_embedding_dim=4,
        rnn_hidden_dim=8,
        rnn_num_layers=1,
        rnn_dropout=0.0,
        attention_dim=4,
        dropout=0.0,
        encoder_chunk_size=3,
        mil_pool_temperature=1.0,
        enable_gradient_checkpointing=False,
        load_pretrained_model=None,
    )


def test_vertebra_mil_model_and_loss_shapes():
    model = Net(_model_cfg())
    model.set_criterion(VertebraMILLoss({"mil_weight": 0.25}))
    batch = {
        "x": torch.zeros(2, 4, 5, 32, 32),
        "y": torch.tensor([[0.0], [1.0]]),
        "level_index": torch.tensor([0, 6]),
        "metadata": torch.zeros(2, 4),
    }
    output = model(batch, return_loss=True)
    assert output["logits"].shape == (2, 1)
    assert output["slice_logits"].shape == (2, 5, 1)
    assert output["attention"].shape == (2, 5)
    assert torch.allclose(output["attention"].sum(1), torch.ones(2))
    assert output["loss"].ndim == 0


def test_vertebra_mil_ignores_non_image_inputs_by_default():
    model = Net(_model_cfg()).eval()
    batch = {
        "x": torch.randn(2, 4, 5, 32, 32),
        "level_index": torch.tensor([0, 6]),
        "metadata": torch.zeros(2, 4),
    }
    with torch.no_grad():
        baseline = model(batch)["logits"]
        batch["metadata"] = torch.randn(2, 4) * 100
        batch["level_index"] = torch.tensor([6, 0])
        changed = model(batch)["logits"]
    assert torch.equal(baseline, changed)


def test_vertebra_mil_skips_padded_slices_and_masks_pooling():
    model = Net(_model_cfg())
    seen_batch_sizes = []
    handle = model.backbone.register_forward_pre_hook(
        lambda _module, inputs: seen_batch_sizes.append(inputs[0].shape[0])
    )
    batch = {
        "x": torch.zeros(2, 4, 5, 32, 32),
        "y": torch.tensor([[0.0], [1.0]]),
        "level_index": torch.tensor([0, 6]),
        "metadata": torch.zeros(2, 4),
        "mask": torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        ),
    }
    output = model(batch)
    handle.remove()
    assert sum(seen_batch_sizes) == 8
    assert torch.count_nonzero(output["attention"][1, 3:]) == 0
    assert torch.allclose(output["attention"].sum(1), torch.ones(2))


def test_multiclass_segmentation_loss_backpropagates():
    criterion = DiceCrossEntropyLoss({})
    logits = torch.randn(2, 8, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 8, (2, 4, 4, 4))
    output = criterion({"logits": logits}, {"y": target})
    output["loss"].backward()
    assert logits.grad is not None


def test_runtime_crop_resampling_preserves_unmagnified_content_scale():
    image = np.zeros((4, 12, 12), dtype=np.float32)
    mask = np.zeros_like(image)
    image[:, 4:8, 3:9] = 10
    mask[:, 4:8, 3:9] = 1
    resized_image, resized_mask = resample_native_crop(
        image,
        mask,
        source_height=4,
        source_width=6,
        axial_scale=1.0,
        output_shape=(6, 10, 10),
    )
    assert resized_image.shape == (6, 10, 10)
    assert resized_mask.shape == (6, 10, 10)
    assert np.allclose(resized_image[:, 3:7, 2:8], 10)
    assert np.allclose(resized_mask[:, 3:7, 2:8], 1)
    assert np.count_nonzero(resized_image[:, :3]) == 0


def test_localization_common_grid_metrics_flatten_slice_positions():
    target = torch.zeros(2, 3, 4, 4, dtype=torch.bool)
    target[0, 1, 1:3, 1:3] = True
    target[1, 2, 0:2, 0:2] = True
    prediction = torch.full((2, 3, 4, 4), 0.05)
    prediction[target] = 0.9
    accumulator = _spatial_accumulator()
    _update_spatial_metrics(accumulator, prediction, target, threshold=0.3)
    metrics = _compute_spatial_metrics(accumulator)
    assert metrics["positions"] == 6
    assert metrics["positive_positions"] == 2
    assert metrics["common_grid_auc"] == pytest.approx(1.0)
    assert metrics["pointing_accuracy"] == pytest.approx(1.0)
