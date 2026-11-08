from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate

from skp.datasets._utils import filter_by_mode


class Dataset(TorchDataset):
    """Series dataset backed by 7-by-32 frozen MIL feature sequences."""

    def __init__(self, cfg, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.collate_fn = default_collate
        path = Path(str(cfg.feature_cache_template).format(fold=int(cfg.fold)))
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fold-specific depth feature cache: {path}. Run "
                "cspine-extract-series-features --representation depth first."
            )
        with np.load(path, allow_pickle=False) as cache:
            required = {"features", "valid_mask", "targets", "fold", "study_uid"}
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(f"Feature cache {path} is missing {sorted(missing)}.")
            features = cache["features"]
            valid_mask = cache["valid_mask"]
            targets = cache["targets"]
            folds = cache["fold"]
            study_uid = cache["study_uid"]
        expected_depth = int(cfg.series_tokens_per_level)
        if features.ndim != 4 or features.shape[1:3] != (7, expected_depth):
            raise ValueError(
                "Expected feature shape "
                f"(N,7,{expected_depth},F), got {features.shape}."
            )
        if valid_mask.shape != features.shape[:3]:
            raise ValueError("valid_mask must match the first three feature dimensions.")
        if targets.shape != (len(features), 8):
            raise ValueError(f"Expected target shape {(len(features), 8)}, got {targets.shape}.")
        if len(folds) != len(features) or len(study_uid) != len(features):
            raise ValueError("Cache identifiers do not align with its features.")

        import pandas as pd

        selected = filter_by_mode(
            pd.DataFrame({"fold": folds, "row_index": np.arange(len(folds))}),
            cfg,
            mode,
        )["row_index"].to_numpy()
        self.features = features[selected]
        self.valid_mask = valid_mask[selected]
        self.targets = targets[selected]
        self.study_uid = study_uid[selected]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "x": torch.from_numpy(self.features[index].astype(np.float32)),
            "mask": torch.from_numpy(self.valid_mask[index].astype(bool)),
            "y": torch.from_numpy(self.targets[index].astype(np.float32)),
            "index": torch.tensor(index),
            "study_uid": str(self.study_uid[index]),
        }
