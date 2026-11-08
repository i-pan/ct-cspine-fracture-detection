from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate

from skp.datasets._utils import filter_by_mode, get_collate_fn, read_annotations


def resample_native_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    axial_scale: float,
    output_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a native-depth crop while preserving the physical crop scale.

    Native crop files are centered in a 384-pixel canvas and never magnified.
    Extracting that fitted content before resizing reproduces the crop builder's
    fit-without-magnification behavior for another axial resolution without
    materializing a second copy of the dataset.
    """
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("image and mask must have matching (D, H, W) shapes.")
    target_depth, target_height, target_width = map(int, output_shape)
    if min(output_shape) <= 0:
        raise ValueError(f"output_shape must be positive, got {output_shape}.")

    fitted_height = min(image.shape[1], max(1, round(source_height * axial_scale)))
    fitted_width = min(image.shape[2], max(1, round(source_width * axial_scale)))
    height_start = (image.shape[1] - fitted_height) // 2
    width_start = (image.shape[2] - fitted_width) // 2
    content = np.stack(
        [
            image[
                :,
                height_start : height_start + fitted_height,
                width_start : width_start + fitted_width,
            ],
            mask[
                :,
                height_start : height_start + fitted_height,
                width_start : width_start + fitted_width,
            ],
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    target_scale = min(
        1.0,
        target_height / float(source_height),
        target_width / float(source_width),
    )
    resized_height = max(1, round(source_height * target_scale))
    resized_width = max(1, round(source_width * target_scale))
    resized = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(content))[None],
        size=(target_depth, resized_height, resized_width),
        mode="trilinear",
        align_corners=False,
    )[0].numpy()
    output = np.zeros((2, target_depth, target_height, target_width), np.float32)
    height_start = (target_height - resized_height) // 2
    width_start = (target_width - resized_width) // 2
    output[
        :,
        :,
        height_start : height_start + resized_height,
        width_start : width_start + resized_width,
    ] = resized
    return output[0], output[1]


def resample_tight_original_x3d_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    output_shape: tuple[int, int, int] = (64, 288, 288),
    mask_threshold: float = 0.4,
) -> np.ndarray:
    """Approximate the original solution's tight chunk and resize contract.

    The competition pipeline cropped to the thresholded segmentation bounds,
    selected/repeated complete slices with nearest-neighbor indexing, and then
    stretched the axial crop to the requested dimensions without preserving
    aspect ratio.
    """
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("image and mask must have matching (D, H, W) shapes.")
    threshold = float(mask_threshold)
    coordinates = np.argwhere(mask >= threshold)
    while coordinates.size == 0 and threshold > 0.1:
        threshold = round(threshold - 0.1, 1)
        coordinates = np.argwhere(mask >= threshold)
    if coordinates.size == 0:
        raise ValueError(
            f"mask has no voxels at or above threshold {threshold}."
        )
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    tight = image[
        lower[0] : upper[0],
        lower[1] : upper[1],
        lower[2] : upper[2],
    ]
    target_depth, target_height, target_width = map(int, output_shape)
    # This is deliberately scipy's order-0 zoom over the integer index grid,
    # matching NumpyChunkDataset.adjust_z_dimension in the original pipeline.
    depth_indices = zoom(
        np.arange(len(tight)),
        float(target_depth) / len(tight),
        order=0,
        prefilter=False,
    )
    tight = tight[depth_indices].astype(np.float32, copy=False)
    # Resize3D used scipy order-1 zoom and ignored aspect ratio. Depth is
    # already exact here, so its depth scale factor is one.
    resized = zoom(
        tight,
        (
            1.0,
            target_height / tight.shape[1],
            target_width / tight.shape[2],
        ),
        order=1,
        prefilter=False,
    )
    if resized.shape != (target_depth, target_height, target_width):
        raise RuntimeError(
            "scipy zoom returned an unexpected shape: "
            f"{resized.shape} != {(target_depth, target_height, target_width)}."
        )
    return resized.astype(np.float32, copy=False)


def resample_close_square_mil_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    output_shape: tuple[int, int, int] = (32, 384, 384),
    mask_threshold: float = 0.4,
    padding_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a close square MIL crop with direction-aware depth resampling.

    Depth is downsampled with trilinear interpolation and upsampled with
    nearest-neighbor repetition. The axial crop is square before its linear
    resize, so no aspect-ratio distortion is introduced.
    """
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("image and mask must have matching (D, H, W) shapes.")
    target_depth, target_height, target_width = map(int, output_shape)
    if target_height != target_width:
        raise ValueError("close square MIL crops require equal output H and W.")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be nonnegative.")

    depth_start, depth_stop, height_start, width_start, side = (
        close_square_mil_geometry(
            mask,
            mask_threshold=mask_threshold,
            padding_fraction=padding_fraction,
        )
    )
    crop = np.stack(
        [
            image[
                depth_start:depth_stop,
                height_start : height_start + side,
                width_start : width_start + side,
            ],
            mask[
                depth_start:depth_stop,
                height_start : height_start + side,
                width_start : width_start + side,
            ],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    crop_tensor = torch.from_numpy(np.ascontiguousarray(crop))[None]
    if crop.shape[1] > target_depth:
        crop_tensor = F.interpolate(
            crop_tensor,
            size=(target_depth, side, side),
            mode="trilinear",
            align_corners=False,
        )
    elif crop.shape[1] < target_depth:
        crop_tensor = F.interpolate(
            crop_tensor,
            size=(target_depth, side, side),
            mode="nearest",
        )
    if side != target_height:
        crop_tensor = F.interpolate(
            crop_tensor,
            size=(target_depth, target_height, target_width),
            mode="trilinear",
            align_corners=False,
        )
    output = crop_tensor[0].numpy()
    return output[0], output[1]


def close_square_mil_geometry(
    mask: np.ndarray,
    *,
    mask_threshold: float = 0.4,
    padding_fraction: float = 0.05,
) -> tuple[int, int, int, int, int]:
    """Return the exact depth and axial bounds used by close-square MIL crops."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a (D,H,W) mask, got {mask.shape}.")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be nonnegative.")
    threshold = float(mask_threshold)
    coordinates = np.argwhere(mask >= threshold)
    while coordinates.size == 0 and threshold > 0.1:
        threshold = round(threshold - 0.1, 1)
        coordinates = np.argwhere(mask >= threshold)
    if coordinates.size == 0:
        raise ValueError(
            f"mask has no voxels at or above threshold {threshold}."
        )

    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    height_extent = int(upper[1] - lower[1])
    width_extent = int(upper[2] - lower[2])
    side = int(
        np.ceil(max(height_extent, width_extent) * (1.0 + 2.0 * padding_fraction))
    )
    side = min(side, mask.shape[1], mask.shape[2])

    def square_start(low: int, high: int, size: int, limit: int) -> int:
        center = (low + high) / 2.0
        return min(max(int(round(center - size / 2.0)), 0), limit - size)

    height_start = square_start(
        int(lower[1]), int(upper[1]), side, mask.shape[1]
    )
    width_start = square_start(
        int(lower[2]), int(upper[2]), side, mask.shape[2]
    )
    return int(lower[0]), int(upper[0]), height_start, width_start, side


def variable_depth_collate(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Right-pad slice sequences while retaining a per-sample validity mask."""
    max_depth = max(sample["x"].shape[1] for sample in samples)
    padded_samples = []
    for sample in samples:
        depth = sample["x"].shape[1]
        padded = sample["x"].new_zeros(
            sample["x"].shape[0],
            max_depth,
            sample["x"].shape[2],
            sample["x"].shape[3],
        )
        padded[:, :depth] = sample["x"]
        item = dict(sample)
        item["x"] = padded
        item["mask"] = torch.arange(max_depth) < depth
        padded_samples.append(item)
    return default_collate(padded_samples)


class Dataset(TorchDataset):
    def __init__(self, cfg, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.df = filter_by_mode(read_annotations(cfg), cfg, mode).reset_index(drop=True)
        collate_fn = variable_depth_collate if cfg.get("variable_depth") else default_collate
        self.collate_fn = get_collate_fn(
            mode,
            train_collate_fn=collate_fn,
            eval_collate_fn=collate_fn,
        )
        targets = self.df["target"].to_numpy(dtype=np.int64)
        counts = np.bincount(targets, minlength=2).astype(np.float64)
        class_weights = len(targets) / np.maximum(2 * counts, 1)
        self.sampling_weights = class_weights[targets]

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _make_channels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        previous = np.concatenate([image[:1], image[:-1]], axis=0)
        following = np.concatenate([image[1:], image[-1:]], axis=0)
        return np.stack([previous, image, following, mask * 255.0], axis=0)

    @staticmethod
    def _make_spaced_stack(
        image: np.ndarray,
        mask: np.ndarray,
        *,
        num_positions: int,
        stack_size: int,
    ) -> np.ndarray:
        if num_positions <= 0 or stack_size <= 0 or stack_size % 2 == 0:
            raise ValueError("num_positions and odd stack_size must be positive.")
        total_slices = num_positions * stack_size
        indices = np.linspace(
            0,
            len(image) - 1,
            total_slices,
        ).round().astype(int)
        grouped = image[indices].reshape(
            num_positions,
            stack_size,
            image.shape[1],
            image.shape[2],
        )
        grouped = grouped.transpose(1, 0, 2, 3)
        center_indices = indices.reshape(num_positions, stack_size)[
            :,
            stack_size // 2,
        ]
        return np.concatenate(
            [grouped, mask[center_indices][None] * 255.0],
            axis=0,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with np.load(row.crop_path) as data:
            image = data["image"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
        runtime_output_shape = self.cfg.get("runtime_output_shape")
        if runtime_output_shape:
            image, mask = resample_native_crop(
                image,
                mask,
                source_height=int(row.source_height),
                source_width=int(row.source_width),
                axial_scale=float(row.axial_scale),
                output_shape=tuple(runtime_output_shape),
            )
        close_square_shape = self.cfg.get("close_square_mil_output_shape")
        if close_square_shape:
            image, mask = resample_close_square_mil_crop(
                image,
                mask,
                output_shape=tuple(close_square_shape),
                mask_threshold=float(
                    self.cfg.get("close_square_mil_mask_threshold", 0.4)
                ),
                padding_fraction=float(
                    self.cfg.get("close_square_mil_padding_fraction", 0.05)
                ),
            )
        original_x3d_shape = self.cfg.get("original_x3d_output_shape")
        if original_x3d_shape:
            image = resample_tight_original_x3d_crop(
                image,
                mask,
                output_shape=tuple(original_x3d_shape),
                mask_threshold=float(self.cfg.get("original_x3d_mask_threshold", 0.4)),
            )
            mask = np.zeros_like(image)
        if self.mode == "train" and self.cfg.get("original_x3d_augmentation"):
            if np.random.random() < 0.5:
                image, mask = image[::-1], mask[::-1]
            if np.random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if np.random.random() < 0.5:
                image, mask = image[:, :, ::-1], mask[:, :, ::-1]
        elif self.mode == "train":
            if np.random.random() < 0.5:
                image, mask = image[:, :, ::-1], mask[:, :, ::-1]
            if np.random.random() < 0.5:
                image, mask = image[:, ::-1, :], mask[:, ::-1, :]
            rotations = np.random.randint(0, 4)
            image = np.rot90(image, rotations, axes=(1, 2))
            mask = np.rot90(mask, rotations, axes=(1, 2))
            image = np.clip(
                image * np.random.uniform(0.9, 1.1)
                + np.random.uniform(-8.0, 8.0),
                0,
                255,
            )
        if self.cfg.get("volume_3d_input", False):
            if self.cfg.get("ct_only_3d_input", False):
                x = image[None]
            else:
                x = np.stack([image, mask * 255.0], axis=0)
        elif self.cfg.get("spaced_stack_size"):
            x = self._make_spaced_stack(
                image,
                mask,
                num_positions=int(self.cfg.num_slices),
                stack_size=int(self.cfg.spaced_stack_size),
            )
        else:
            x = self._make_channels(image, mask)
        x = np.ascontiguousarray(x)
        metadata = np.array(
            [
                np.log1p(float(row.source_depth)),
                np.log1p(float(row.source_height)),
                np.log1p(float(row.source_width)),
                float(row.mean_z_spacing),
            ],
            dtype=np.float32,
        )
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor([float(row.target)]),
            "level_index": torch.tensor(int(row.level_index)),
            "metadata": torch.from_numpy(metadata),
            "index": torch.tensor(index),
        }
