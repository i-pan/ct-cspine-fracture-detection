from pathlib import Path

import numpy as np
import pytest

from skp.cspine.artifacts import read_artifact_manifest
from skp.cspine.pipeline import (
    DICOMSeries,
    FracturePredictionResult,
    LevelCropGeometry,
    LevelLocalization,
    _make_adjacent_channels,
    _make_stack3_channels,
    _resampled_source_indices,
)


def _series(depth: int = 3) -> DICOMSeries:
    return DICOMSeries(
        volume=np.zeros((depth, 8, 8), dtype=np.uint8),
        study_uid="study",
        series_uid="series",
        instance_numbers=np.arange(depth, dtype=np.int32),
        image_position_z=np.arange(depth, dtype=np.float32),
        pixel_spacing=(1.0, 1.0),
        slice_spacing=1.0,
    )


def test_dicom_series_requires_aligned_uint8_volume():
    assert _series().volume.shape == (3, 8, 8)
    with pytest.raises(ValueError, match="uint8"):
        DICOMSeries(
            volume=np.zeros((3, 8, 8), dtype=np.float32),
            study_uid="study",
            series_uid="series",
            instance_numbers=np.arange(3),
            image_position_z=np.arange(3),
            pixel_spacing=(1.0, 1.0),
            slice_spacing=1.0,
        )


def test_channel_builders_preserve_adjacent_and_nonoverlapping_triplets():
    image = np.arange(96, dtype=np.float32)[:, None, None]
    mask = np.ones_like(image)
    adjacent = _make_adjacent_channels(image[:32], mask[:32])
    assert adjacent[:, 0, 0, 0].tolist() == [0, 0, 1, 255]
    stack3 = _make_stack3_channels(image, mask)
    assert stack3[:3, 0, 0, 0].tolist() == [0, 1, 2]
    assert stack3[:3, 1, 0, 0].tolist() == [3, 4, 5]
    assert stack3[3, 0, 0, 0] == 255


def test_direction_aware_depth_index_policy():
    assert _resampled_source_indices(2, 4).tolist() == [0, 0, 1, 1]
    assert _resampled_source_indices(4, 2).tolist() == [1, 3]


def test_prediction_save_keeps_heatmap_out_of_json(tmp_path: Path):
    geometry = LevelCropGeometry(
        level="C2",
        native_bounds=((0, 2), (1, 5), (2, 6)),
        source_shape=(2, 4, 4),
        axial_scale=1.0,
        fitted_canvas_start=(0, 0),
        close_square=(0, 2, 0, 0, 4),
        native_slice_indices=tuple(range(96)),
        native_instance_numbers=tuple(range(96)),
    )
    result = FracturePredictionResult(
        probabilities={"overall": 0.8, **{f"C{i}": 0.1 for i in range(1, 8)}},
        architecture_probabilities={},
        localizations={
            "C2": LevelLocalization(
                level="C2",
                heatmap=np.zeros((96, 4, 4), dtype=np.float32),
                slice_probabilities=np.zeros(96, dtype=np.float32),
                geometry=geometry,
            )
        },
        missing_levels=(),
    )
    result.save(tmp_path)
    assert (tmp_path / "prediction.json").is_file()
    with np.load(tmp_path / "C2_localization.npz") as saved:
        assert saved["heatmap"].shape == (96, 4, 4)
    assert '"heatmap":' not in (tmp_path / "prediction.json").read_text()


def test_localization_can_be_restored_to_native_slice_coordinates():
    geometry = LevelCropGeometry(
        level="C2",
        native_bounds=((1, 3), (2, 6), (3, 7)),
        source_shape=(2, 4, 4),
        axial_scale=1.0,
        fitted_canvas_start=(190, 190),
        close_square=(0, 2, 190, 190, 4),
        native_slice_indices=(1, 1, 2),
        native_instance_numbers=(11, 11, 12),
    )
    heatmap = np.stack(
        [
            np.full((4, 4), 0.2, np.float32),
            np.full((4, 4), 0.8, np.float32),
            np.full((4, 4), 0.4, np.float32),
        ]
    )
    localization = LevelLocalization(
        level="C2",
        heatmap=heatmap,
        slice_probabilities=heatmap.reshape(3, -1).max(1),
        geometry=geometry,
    )

    restored = list(localization.iter_native_heatmaps((4, 8, 9)))

    assert [(item[0], item[1]) for item in restored] == [(1, 11), (2, 12)]
    assert np.allclose(restored[0][2][2:6, 3:7], 0.8)
    assert np.count_nonzero(restored[0][2][:2]) == 0
    assert np.allclose(restored[1][2][2:6, 3:7], 0.4)


def test_artifact_manifest_is_typed_and_versioned(tmp_path: Path):
    (tmp_path / "cspine_model.json").write_text(
        '{"format_version": 1, "model_type": "cspine-segmentation"}'
    )
    assert (
        read_artifact_manifest(tmp_path, "cspine-segmentation")["format_version"] == 1
    )
    with pytest.raises(ValueError, match="model_type"):
        read_artifact_manifest(tmp_path, "cspine-fracture-classification")
