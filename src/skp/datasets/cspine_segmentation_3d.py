from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from skp.datasets._utils import filter_by_mode, get_collate_fn, read_annotations


class Dataset(TorchDataset):
    def __init__(self, cfg, mode: str):
        self.cfg = cfg
        self.mode = mode
        annotation_cfg = cfg
        validation_annotations = cfg.get("validation_annotations_file")
        if mode != "train" and validation_annotations:
            annotation_cfg = copy.copy(cfg)
            annotation_cfg.annotations_file = validation_annotations
        self.df = filter_by_mode(
            read_annotations(annotation_cfg), cfg, mode
        ).reset_index(drop=True)
        self.collate_fn = get_collate_fn(mode)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with np.load(row.volume_path) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.int64)
        if self.mode == "train":
            if np.random.random() < 0.5:
                image, mask = image[:, :, ::-1], mask[:, :, ::-1]
            if np.random.random() < 0.5:
                image, mask = image[:, ::-1, :], mask[:, ::-1, :]
            rotations = np.random.randint(0, 4)
            image = np.rot90(image, rotations, axes=(1, 2))
            mask = np.rot90(mask, rotations, axes=(1, 2))
            gain = np.random.uniform(0.9, 1.1)
            offset = np.random.uniform(-8.0, 8.0)
            image = np.clip(image * gain + offset, 0, 255)
        image = np.ascontiguousarray(image[None])
        mask = np.ascontiguousarray(mask)
        return {
            "x": torch.from_numpy(image).float(),
            "y": torch.from_numpy(mask).long(),
            "index": torch.tensor(index),
        }
