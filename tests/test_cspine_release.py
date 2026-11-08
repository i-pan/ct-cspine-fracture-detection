from pathlib import Path

import numpy as np
import pytest
import torch

from skp.configs.cspine_segmentation_x3d_s import cfg as segmentation_cfg
from skp.configs.cspine_vertebra_mil_maxvit_tiny import cfg as maxvit_cfg
from skp.cspine.artifacts import checkpoint_model_state
from skp.cspine.hf_export import _export_checkpoint
from skp.cspine.release_manifest import component_run_root, load_release_manifest
from skp.cspine.series_oof import align_oof_members, probability_metrics
from skp.cspine.segmentation_state_evaluation import multiclass_dice_by_sample
from skp.models.utils import state_dict_sha256, verify_state_dict_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_release_manifest_selects_complete_four_head_ensemble():
    manifest = load_release_manifest(PROJECT_ROOT / "release" / "final-v1.json")
    selected = manifest["selected_pipeline"]
    assert selected["segmentation"]["checkpoint_state"] == "ema"
    assert selected["segmentation"]["training_crop_checkpoint_state"] == "raw"
    assert selected["series_ensemble"]["members"] == [
        "d32.series_7x1",
        "d32.series_7x32",
        "3x32.series_7x1",
        "3x32.series_7x32",
    ]
    assert selected["families"]["3x32"]["mil"]["fold_overrides"]["4"]
    fold4 = component_run_root(
        Path("/experiments"), selected["families"]["3x32"]["mil"], fold=4
    )
    assert "loaderfix" in str(fold4)
    assert selected["families"]["d32"]["crop_tensor"] == "d32"
    assert selected["families"]["3x32"]["crop_tensor"] == "stack3"
    initializations = manifest["pretrained_initialization"]
    assert (
        segmentation_cfg.pretrained_state_sha256
        == initializations["x3d_s"]["canonical_state_sha256"]
    )
    assert (
        maxvit_cfg.pretrained_state_sha256
        == initializations["maxvit_tiny_tf_384.in1k_4ch"][
            "canonical_state_sha256"
        ]
    )


def test_checkpoint_export_refuses_to_reuse_stale_weights(tmp_path: Path):
    output_path = tmp_path / "fold0.safetensors"
    output_path.write_bytes(b"old weights")
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        _export_checkpoint(
            tmp_path / "new.ckpt",
            output_path,
            use_ema=True,
            overwrite=False,
        )


def test_checkpoint_state_selection_distinguishes_raw_and_ema(tmp_path: Path):
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {"model.weight": torch.tensor([1.0])},
            "callbacks": {
                "EMACallback": {
                    "ema_state_dict": {"model.weight": torch.tensor([2.0])}
                }
            },
        },
        checkpoint,
    )

    raw = checkpoint_model_state(checkpoint, use_ema=False)
    ema = checkpoint_model_state(checkpoint, use_ema=True)

    assert raw["weight"].item() == 1.0
    assert ema["weight"].item() == 2.0

    plain_checkpoint = tmp_path / "plain.pt"
    torch.save({"weight": torch.tensor([3.0])}, plain_checkpoint)
    plain = checkpoint_model_state(plain_checkpoint, use_ema=False)
    assert plain["weight"].item() == 3.0


def test_pretrained_state_hash_is_stable_and_fails_closed():
    state = {
        "b": torch.tensor([2.0]),
        "a": torch.tensor([[1, 2]], dtype=torch.int64),
    }
    expected = state_dict_sha256(state)
    assert expected == state_dict_sha256(dict(reversed(list(state.items()))))
    verify_state_dict_sha256(state, expected, name="fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_state_dict_sha256(state, "0" * 64, name="fixture")


def test_oof_members_align_by_uid_and_reject_target_mismatch():
    uids = np.array(["study-b", "study-a"])
    targets = np.array([[1] * 8, [0] * 8], dtype=np.float32)
    first = np.array([[0.8] * 8, [0.2] * 8], dtype=np.float32)
    second = np.array([[0.1] * 8, [0.9] * 8], dtype=np.float32)
    aligned_uids, aligned_targets, aligned = align_oof_members(
        {
            "first": (uids, targets, first),
            "second": (uids[::-1], targets[::-1], second),
        }
    )
    assert aligned_uids.tolist() == ["study-a", "study-b"]
    assert aligned_targets[:, 0].tolist() == [0.0, 1.0]
    assert aligned["second"][:, 0].tolist() == pytest.approx([0.1, 0.9])

    bad_targets = targets[::-1].copy()
    bad_targets[0, 0] = 1
    with pytest.raises(ValueError, match="Target mismatch"):
        align_oof_members(
            {
                "first": (uids, targets, first),
                "bad": (uids[::-1], bad_targets, second),
            }
        )


def test_series_probability_metrics_use_all_eight_outputs():
    targets = np.tile([[0.0], [1.0]], (1, 8)).astype(np.float32)
    probabilities = np.tile([[0.1], [0.9]], (1, 8)).astype(np.float32)
    metrics = probability_metrics(targets, probabilities)
    assert metrics["competition_log_loss"] == pytest.approx(-np.log(0.9))
    assert metrics["auc_overall"] == 1.0
    assert metrics["auc_levels_mean"] == 1.0


def test_segmentation_state_dice_ignores_absent_sample_classes():
    targets = np.array([[[1, 1]], [[2, 2]]], dtype=np.uint8)
    predictions = np.array([[[1, 0]], [[2, 2]]], dtype=np.uint8)
    metrics = multiclass_dice_by_sample(predictions, targets, classes=range(1, 3))
    assert metrics["dice_c1"] == pytest.approx(2 / 3)
    assert metrics["dice_c2"] == 1.0
    assert metrics["dice_mean"] == pytest.approx(5 / 6)
