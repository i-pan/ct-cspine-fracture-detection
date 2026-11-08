from __future__ import annotations

import json
import os
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from skp.cspine.constants import LEVELS


def resize_volume(
    volume: np.ndarray,
    output_shape: tuple[int, int, int],
    *,
    mode: str,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(volume)).float()[None, None]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=output_shape, mode=mode)
    else:
        resized = F.interpolate(
            tensor, size=output_shape, mode=mode, align_corners=False
        )
    return resized[0, 0].numpy()


def resize_depth_and_fit_axial(
    volume: np.ndarray,
    output_shape: tuple[int, int, int],
    *,
    mode: str,
) -> tuple[np.ndarray, float]:
    """Resize depth and fit the axial plane without magnifying native pixels."""
    depth, target_height, target_width = output_shape
    _, source_height, source_width = volume.shape
    axial_scale = min(
        1.0,
        target_height / source_height,
        target_width / source_width,
    )
    resized_height = max(1, round(source_height * axial_scale))
    resized_width = max(1, round(source_width * axial_scale))
    resized = resize_volume(
        volume,
        (depth, resized_height, resized_width),
        mode=mode,
    )
    output = np.zeros(output_shape, dtype=resized.dtype)
    height_start = (target_height - resized_height) // 2
    width_start = (target_width - resized_width) // 2
    output[
        :,
        height_start : height_start + resized_height,
        width_start : width_start + resized_width,
    ] = resized
    return output, float(axial_scale)


def vertebra_bounds(
    mask: np.ndarray, level_index: int, padding_fraction: float = 0.15
) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask == level_index)
    if coordinates.size == 0:
        raise ValueError(f"Mask does not contain vertebra label {level_index}.")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    extent = upper - lower
    padding = np.maximum(np.ceil(extent * padding_fraction).astype(int), 2)
    lower = np.maximum(lower - padding, 0)
    upper = np.minimum(upper + padding, mask.shape)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper))


def map_bounds_to_shape(
    bounds: tuple[slice, slice, slice],
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    mapped = []
    for bound, source, target in zip(bounds, source_shape, target_shape):
        start = int(np.floor(bound.start * target / source))
        stop = int(np.ceil(bound.stop * target / source))
        mapped.append(slice(max(start, 0), min(stop, target)))
    return tuple(mapped)


def load_png_volume(study_slices: pd.DataFrame) -> np.ndarray:
    ordered = study_slices.sort_values("slice_index")
    images = [
        cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in ordered["png_path"]
    ]
    if any(image is None for image in images):
        raise FileNotFoundError("At least one PNG could not be decoded.")
    shapes = {image.shape for image in images}
    if len(shapes) != 1:
        raise ValueError(f"All slices in one study must share dimensions, got {shapes}")
    return np.stack(images)


def build_vertebra_crops(
    slices: pd.DataFrame,
    studies: pd.DataFrame,
    mask_root: str | Path,
    output_root: str | Path,
    *,
    output_shape: tuple[int, int, int] = (32, 384, 384),
    padding_fraction: float = 0.25,
    allow_missing: bool = False,
    study_folds: set[int] | None = None,
    crop_subdir: str = "vertebra_crops",
    manifest_name: str = "vertebra_samples.parquet",
    variable_depth: bool = False,
    exclude_unresolved_qc: bool = False,
    qc_teacher_annotations: str | Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build fixed-shape crops from fold-safe ``<study_uid>.npy`` masks."""
    mask_root = Path(mask_root)
    output_root = Path(output_root)
    crop_root = output_root / crop_subdir
    rows = []
    missing_levels = []
    studies_by_uid = studies.set_index("StudyInstanceUID")
    teacher_by_uid = {}
    if qc_teacher_annotations:
        teacher_annotations = pd.read_parquet(qc_teacher_annotations)
        teacher_by_uid = teacher_annotations.set_index("StudyInstanceUID")[
            "volume_path"
        ].to_dict()
    for study_uid, study_slices in slices.query("split == 'train'").groupby(
        "study_uid", sort=False
    ):
        study = studies_by_uid.loc[study_uid]
        if study_folds is not None and int(study["fold"]) not in study_folds:
            continue
        unresolved_qc = False
        if exclude_unresolved_qc or teacher_by_uid:
            qc_path = mask_root / "_qc" / f"{study_uid}.json"
            if qc_path.is_file():
                qc = json.loads(qc_path.read_text())
                unresolved_qc = bool(qc.get("final_anomaly_levels"))
                if unresolved_qc and study_uid not in teacher_by_uid:
                    continue
        mask_path = mask_root / f"{study_uid}.npy"
        if not mask_path.exists():
            continue
        volume = load_png_volume(study_slices)
        if unresolved_qc:
            with np.load(teacher_by_uid[study_uid]) as teacher_volume:
                mask = teacher_volume["mask"].astype(np.uint8)
            mask_source = "teacher_qc_fallback"
        else:
            mask = np.load(mask_path)
            mask_source = "student"
        ordered = study_slices.sort_values("slice_index")
        for level_index, level in enumerate(LEVELS, start=1):
            try:
                mask_bounds = vertebra_bounds(mask, level_index, padding_fraction)
            except ValueError:
                missing_levels.append((study_uid, level))
                continue
            bounds = map_bounds_to_shape(mask_bounds, mask.shape, volume.shape)
            crop = volume[bounds]
            source_rows = ordered.iloc[bounds[0]]
            target_depth = (
                min(crop.shape[0], output_shape[0])
                if variable_depth
                else output_shape[0]
            )
            sample_shape = (target_depth, output_shape[1], output_shape[2])
            level_dir = crop_root / study_uid
            level_dir.mkdir(parents=True, exist_ok=True)
            crop_path = level_dir / f"{level}.npz"
            axial_scale = min(
                1.0,
                output_shape[1] / crop.shape[1],
                output_shape[2] / crop.shape[2],
            )
            valid_existing = False
            if crop_path.is_file() and not overwrite:
                try:
                    with np.load(crop_path) as existing:
                        valid_existing = (
                            existing["image"].shape == sample_shape
                            and existing["mask"].shape == sample_shape
                        )
                except (
                    OSError,
                    ValueError,
                    EOFError,
                    zipfile.BadZipFile,
                    zlib.error,
                ):
                    valid_existing = False
            if not valid_existing:
                mask_crop = (mask[mask_bounds] == level_index).astype(np.float32)
                crop_mask = resize_volume(mask_crop, crop.shape, mode="trilinear")
                resized, axial_scale = resize_depth_and_fit_axial(
                    crop,
                    sample_shape,
                    mode="trilinear",
                )
                resized_mask, _ = resize_depth_and_fit_axial(
                    crop_mask,
                    sample_shape,
                    mode="trilinear",
                )
                temporary_path = crop_path.with_suffix(".npz.tmp")
                with temporary_path.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        image=np.rint(resized).astype(np.uint8),
                        mask=resized_mask.astype(np.float16),
                    )
                os.replace(temporary_path, crop_path)
            depth_map = np.linspace(
                0, len(source_rows) - 1, target_depth
            ).round().astype(int)
            source_instances = source_rows.iloc[depth_map][
                "instance_number"
            ].astype(int).tolist()
            source_z = source_rows.iloc[depth_map][
                "image_position_z"
            ].astype(float).tolist()
            rows.append(
                {
                    "StudyInstanceUID": study_uid,
                    "level": level,
                    "level_index": level_index - 1,
                    "target": int(study[level]),
                    "patient_overall": int(study["patient_overall"]),
                    "fold": int(study["fold"]),
                    "crop_path": str(crop_path.resolve()),
                    "source_instances": json.dumps(source_instances),
                    "source_z": json.dumps(source_z),
                    "source_depth": int(crop.shape[0]),
                    "input_depth": int(target_depth),
                    "source_height": int(crop.shape[1]),
                    "source_width": int(crop.shape[2]),
                    "axial_scale": axial_scale,
                    "mask_source": mask_source,
                    "mean_z_spacing": float(
                        source_rows["actual_z_spacing"].median()
                    ),
                }
            )
    if missing_levels and not allow_missing:
        examples = ", ".join(f"{uid}/{level}" for uid, level in missing_levels[:10])
        raise ValueError(
            f"Predicted masks are missing {len(missing_levels)} vertebrae "
            f"(examples: {examples}). Revisit segmentation or pass allow_missing=True."
        )
    samples = pd.DataFrame(rows)
    manifest_path = output_root / "manifests" / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(manifest_path, index=False)
    return samples
