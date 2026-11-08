from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate

from skp.datasets._utils import filter_by_mode


class Dataset(TorchDataset):
    """Series dataset backed by a fold-specific frozen-MIL feature cache."""

    def __init__(self, cfg, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.collate_fn = default_collate
        path = Path(str(cfg.feature_cache_template).format(fold=int(cfg.fold)))
        if not path.exists():
            raise FileNotFoundError(
                f"Missing fold-specific series feature cache: {path}. Run "
                "cspine-extract-series-features first."
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
        if features.ndim != 3 or features.shape[1] != 7:
            raise ValueError(f"Expected feature shape (N,7,F), got {features.shape}.")
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask must match the first two feature dimensions.")
        if targets.shape != (len(features), 8):
            raise ValueError(f"Expected target shape {(len(features), 8)}, got {targets.shape}.")
        if len(folds) != len(features) or len(study_uid) != len(features):
            raise ValueError("Cache identifiers do not align with its features.")

        # Reuse the framework's explicit fold contract without reading a
        # redundant annotation table. No sampling weights are constructed.
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
