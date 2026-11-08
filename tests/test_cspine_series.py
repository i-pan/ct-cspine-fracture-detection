from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from skp.configs import Config
from skp.cspine.series_features import (
    assemble_series_cache,
    assemble_series_depth_cache,
)
from skp.datasets.cspine_series_depth_features import Dataset as DepthDataset
from skp.datasets.cspine_series_features import Dataset
from skp.losses.cspine import SeriesCompetitionLoss
from skp.models.classification.cspine_series import Net
from skp.models.classification.cspine_series_depth import Net as DepthNet


def test_series_competition_loss_matches_reshaped_weighted_bce():
    logits = torch.tensor(
        [[0.2, -0.5, 0.1, 0.4, -0.2, 0.7, -0.8, 0.3]],
        dtype=torch.float32,
    )
    target = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.float32)
    result = SeriesCompetitionLoss({})({"logits": logits}, {"y": target})
    negative = torch.tensor([7.0] + [1.0] * 7)
    weights = negative * (1 + target[0])
    expected = (
        F.binary_cross_entropy_with_logits(logits[0], target[0], reduction="none")
        * weights
    ).sum() / weights.sum()
    assert result["loss"] == pytest.approx(expected)


def test_assemble_series_cache_orders_levels_and_marks_missing():
    studies = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "patient_overall": [1, 0],
            **{f"C{level}": [int(level == 2), 0] for level in range(1, 8)},
            "fold": [0, 1],
        }
    )
    vertebrae = pd.DataFrame(
        {
            "StudyInstanceUID": ["b", "a", "a"],
            "level_index": [6, 1, 0],
        }
    )
    row_features = np.array([[7, 70], [2, 20], [1, 10]], dtype=np.float16)
    result = assemble_series_cache(studies, vertebrae, row_features)
    assert result["features"].shape == (2, 7, 2)
    assert result["features"][0, 0].tolist() == [1, 10]
    assert result["features"][0, 1].tolist() == [2, 20]
    assert result["features"][1, 6].tolist() == [7, 70]
    assert result["valid_mask"].sum() == 3
    assert result["targets"][0].tolist() == [1, 0, 1, 0, 0, 0, 0, 0]


def test_assemble_series_depth_cache_orders_levels_and_marks_depth():
    studies = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "patient_overall": [1, 0],
            **{f"C{level}": [int(level == 2), 0] for level in range(1, 8)},
            "fold": [0, 1],
        }
    )
    vertebrae = pd.DataFrame(
        {"StudyInstanceUID": ["b", "a"], "level_index": [6, 1]}
    )
    row_features = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    result = assemble_series_depth_cache(studies, vertebrae, row_features)
    assert result["features"].shape == (2, 7, 3, 4)
    assert result["features"][0, 1].tolist() == row_features[1].tolist()
    assert result["features"][1, 6].tolist() == row_features[0].tolist()
    assert result["valid_mask"].sum() == 6


def _series_config(feature_dim: int, use_projection: bool) -> Config:
    cfg = Config()
    cfg.series_feature_dim = feature_dim
    cfg.series_projection_dim = 4
    cfg.use_layernorm_projection = use_projection
    cfg.series_hidden_dim = 3
    cfg.series_num_layers = 1
    cfg.series_attention_dim = 2
    cfg.dropout = 0.0
    return cfg


@pytest.mark.parametrize("use_projection", [True, False])
def test_series_model_outputs_overall_then_seven_levels(use_projection: bool):
    model = Net(_series_config(6, use_projection))
    model.set_criterion(SeriesCompetitionLoss({}))
    batch = {
        "x": torch.randn(2, 7, 6),
        "mask": torch.tensor(
            [[True] * 7, [True, True, True, True, True, True, False]]
        ),
        "y": torch.randint(0, 2, (2, 8)).float(),
    }
    output = model(batch, return_loss=True, return_features=True)
    assert output["logits"].shape == (2, 8)
    assert output["features_seq"].shape == (2, 7, 6)
    assert output["overall_attention"].sum(dim=1).tolist() == pytest.approx([1, 1])
    assert torch.isfinite(output["loss"])


def test_series_dataset_uses_fold_cache_without_sampling_weights(tmp_path: Path):
    path = tmp_path / "fold0.npz"
    np.savez_compressed(
        path,
        features=np.zeros((3, 7, 5), dtype=np.float16),
        valid_mask=np.ones((3, 7), dtype=bool),
        targets=np.zeros((3, 8), dtype=np.uint8),
        fold=np.array([0, 1, 1], dtype=np.int8),
        study_uid=np.array(["a", "b", "c"]),
    )
    cfg = Config()
    cfg.feature_cache_template = str(tmp_path / "fold{fold}.npz")
    cfg.fold = 0
    cfg.split_column = "fold"
    train = Dataset(cfg, "train")
    validation = Dataset(cfg, "val")
    assert len(train) == 2
    assert len(validation) == 1
    assert not hasattr(train, "sampling_weights")
    assert train[0]["x"].shape == (7, 5)


@pytest.mark.parametrize("use_projection", [True, False])
def test_depth_series_model_outputs_overall_then_seven_levels(use_projection: bool):
    cfg = _series_config(6, use_projection)
    cfg.series_tokens_per_level = 4
    model = DepthNet(cfg)
    model.set_criterion(SeriesCompetitionLoss({}))
    batch = {
        "x": torch.randn(2, 7, 4, 6),
        "mask": torch.tensor(
            [
                [[True] * 4] * 7,
                [[True] * 4] * 6 + [[False] * 4],
            ]
        ),
        "y": torch.randint(0, 2, (2, 8)).float(),
    }
    output = model(batch, return_loss=True, return_features=True)
    assert output["logits"].shape == (2, 8)
    assert output["features_seq"].shape == (2, 28, 6)
    assert output["features_level"].shape == (2, 7, 6)
    assert torch.allclose(
        output["level_attention"].sum(dim=2),
        torch.ones(2, 7),
    )
    assert output["overall_attention"].sum(dim=1).tolist() == pytest.approx([1, 1])
    assert torch.isfinite(output["loss"])


def test_depth_series_dataset_uses_7_by_depth_cache(tmp_path: Path):
    path = tmp_path / "fold0.npz"
    np.savez_compressed(
        path,
        features=np.zeros((3, 7, 4, 5), dtype=np.float16),
        valid_mask=np.ones((3, 7, 4), dtype=bool),
        targets=np.zeros((3, 8), dtype=np.uint8),
        fold=np.array([0, 1, 1], dtype=np.int8),
        study_uid=np.array(["a", "b", "c"]),
    )
    cfg = Config()
    cfg.feature_cache_template = str(tmp_path / "fold{fold}.npz")
    cfg.series_tokens_per_level = 4
    cfg.fold = 0
    cfg.split_column = "fold"
    train = DepthDataset(cfg, "train")
    validation = DepthDataset(cfg, "val")
    assert len(train) == 2
    assert len(validation) == 1
    assert not hasattr(train, "sampling_weights")
    assert train[0]["x"].shape == (7, 4, 5)
