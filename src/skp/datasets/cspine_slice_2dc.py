"""Three-adjacent-slice cervical fracture pseudolabel dataset."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset

from skp.datasets._utils import get_collate_fn, get_transforms, read_annotations


class Dataset(TorchDataset):
    def __init__(self, cfg, mode: str):
        self.cfg = cfg
        self.mode = mode
        df = read_annotations(cfg)
        if mode == "train":
            # The experiment intentionally distils all OOF teacher predictions into
            # one deployable model. Manual labels are never used as training targets.
            self.df = df.reset_index(drop=True)
            self.labels = self.df[["teacher_probability"]].to_numpy(np.float32)
        elif mode in {"val", "test"}:
            self.df = df[df["manual_available"]].reset_index(drop=True)
            self.labels = self.df[["manual_target"]].to_numpy(np.float32)
        else:
            self.df = df.reset_index(drop=True)
            self.labels = self.df[["teacher_probability"]].to_numpy(np.float32)

        self.paths = self.df[
            ["png_path_prev", "png_path_center", "png_path_next"]
        ].to_numpy()
        self.transforms = get_transforms(cfg, mode)
        self.collate_fn = get_collate_fn(mode)

        if mode == "train" and cfg.get("sampler") == "WeightedSampler":
            positive = self.labels[:, 0] >= float(
                cfg.get("pseudo_positive_threshold", 0.5)
            )
            positive_count = max(int(positive.sum()), 1)
            negative_count = max(int((~positive).sum()), 1)
            positive_fraction = float(cfg.get("sample_positive_fraction", 0.5))
            self.sampling_weights = np.where(
                positive,
                positive_fraction / positive_count,
                (1.0 - positive_fraction) / negative_count,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        images = []
        for path in self.paths[index]:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            images.append(image)
        image = np.stack(images, axis=-1)
        image = self.transforms(image=image)["image"]
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = rearrange(image, "h w c -> c h w").float()
        return {
            "x": image,
            "y": torch.from_numpy(self.labels[index]).float(),
            "fold": torch.tensor(int(self.df.iloc[index]["fold"])),
            "index": torch.tensor(index),
        }
