from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate

from skp.datasets._utils import filter_by_mode, read_annotations
from skp.datasets.cspine_vertebra_mil import resample_close_square_mil_crop


class Dataset(TorchDataset):
    """One binary fracture heatmap volume per positive vertebral crop."""

    def __init__(self, cfg, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.df = filter_by_mode(read_annotations(cfg), cfg, mode).reset_index(drop=True)
        self.collate_fn = default_collate

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with np.load(row.crop_path) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
        image, mask = resample_close_square_mil_crop(
            image,
            mask,
            output_shape=tuple(self.cfg.close_square_mil_output_shape),
            mask_threshold=float(self.cfg.close_square_mil_mask_threshold),
            padding_fraction=float(self.cfg.close_square_mil_padding_fraction),
        )
        stack_size = int(self.cfg.get("spaced_stack_size") or 0)
        if stack_size:
            if len(image) % stack_size:
                raise ValueError(
                    f"Depth {len(image)} is not divisible by stack size {stack_size}."
                )
            num_positions = len(image) // stack_size
            grouped = image.reshape(
                num_positions, stack_size, image.shape[1], image.shape[2]
            ).transpose(1, 0, 2, 3)
            center = mask[
                np.arange(num_positions) * stack_size + stack_size // 2
            ]
            x = np.concatenate([grouped, center[None] * 255.0], axis=0)
        else:
            previous = np.concatenate([image[:1], image[:-1]], axis=0)
            following = np.concatenate([image[1:], image[-1:]], axis=0)
            x = np.stack([previous, image, following, mask * 255.0], axis=0)

        target_size = int(self.cfg.localization_target_size)
        y = np.zeros((len(image), target_size, target_size), dtype=np.float32)
        scale = target_size / float(self.cfg.image_height)
        for position, boxes in enumerate(json.loads(row.boxes_by_position)):
            for x0, y0, x1, y1 in boxes:
                ix0 = max(0, min(target_size, int(np.floor(x0 * scale))))
                iy0 = max(0, min(target_size, int(np.floor(y0 * scale))))
                ix1 = max(0, min(target_size, int(np.ceil(x1 * scale))))
                iy1 = max(0, min(target_size, int(np.ceil(y1 * scale))))
                if ix1 > ix0 and iy1 > iy0:
                    y[position, iy0:iy1, ix0:ix1] = 1.0
        if stack_size:
            y = y.reshape(num_positions, stack_size, target_size, target_size)
            y = y.transpose(1, 0, 2, 3)
        else:
            y = y[None]

        if self.mode == "train":
            if np.random.random() < 0.5:
                x = x[..., ::-1]
                y = y[..., ::-1]
            if np.random.random() < 0.5:
                x = x[..., ::-1, :]
                y = y[..., ::-1, :]
            rotations = np.random.randint(0, 4)
            x = np.rot90(x, rotations, axes=(-2, -1))
            y = np.rot90(y, rotations, axes=(-2, -1))
            x[:3] = np.clip(
                x[:3] * np.random.uniform(0.9, 1.1)
                + np.random.uniform(-8.0, 8.0),
                0,
                255,
            )

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "y": torch.from_numpy(np.ascontiguousarray(y)).float(),
            "index": torch.tensor(index),
        }
