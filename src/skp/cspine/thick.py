from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skp.cspine.dicom import _write_png_atomic, dicom_to_hu, hu_to_uint8


def _plane_boundaries(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 1 or len(positions) < 2:
        raise ValueError("At least two one-dimensional z positions are required.")
    if not np.all(np.diff(positions) > 0):
        raise ValueError("z positions must be strictly increasing.")
    midpoints = (positions[:-1] + positions[1:]) / 2
    return np.r_[
        positions[0] - (midpoints[0] - positions[0]),
        midpoints,
        positions[-1] + (positions[-1] - midpoints[-1]),
    ]


def slab_weights(
    source_positions: np.ndarray, center: float, thickness: float
) -> np.ndarray:
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    boundaries = _plane_boundaries(source_positions)
    lower, upper = center - thickness / 2, center + thickness / 2
    overlap = np.maximum(
        0.0,
        np.minimum(boundaries[1:], upper) - np.maximum(boundaries[:-1], lower),
    )
    if overlap.sum() == 0:
        nearest = int(np.argmin(np.abs(source_positions - center)))
        overlap[nearest] = 1.0
    return overlap / overlap.sum()


def reconstruct_thick_volume(
    volume_hu: np.ndarray,
    source_positions: np.ndarray,
    *,
    thickness: float,
    interval: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Reconstruct regularly spaced thick slabs using overlap-weighted HU means."""
    volume = np.asarray(volume_hu, dtype=np.float32)
    positions = np.asarray(source_positions, dtype=float)
    if volume.ndim != 3 or volume.shape[0] != len(positions):
        raise ValueError("volume must have shape (len(source_positions), H, W).")
    if positions[0] > positions[-1]:
        positions = positions[::-1]
        volume = volume[::-1]
    interval = thickness if interval is None else interval
    if interval <= 0:
        raise ValueError("interval must be positive.")
    centers = np.arange(positions[0], positions[-1] + interval / 2, interval)
    slabs, contributors = [], []
    for center in centers:
        weights = slab_weights(positions, center, thickness)
        slabs.append(np.tensordot(weights, volume, axes=(0, 0)))
        contributors.append(np.flatnonzero(weights > 0).astype(int).tolist())
    return np.stack(slabs), centers, contributors


def build_thick_series(
    slice_manifest: pd.DataFrame,
    output_root: str | Path,
    *,
    thicknesses: tuple[float, ...] = (2.0, 3.0, 5.0),
    split: str = "train",
    overwrite: bool = False,
) -> pd.DataFrame:
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Thick reconstruction requires pydicom.") from exc

    output_root = Path(output_root)
    records = []
    selected = slice_manifest[slice_manifest["split"].eq(split)]
    for study_uid, group in selected.groupby("study_uid", sort=False):
        ordered = group.sort_values(["image_position_z", "instance_number"])
        datasets = [pydicom.dcmread(path) for path in ordered["dcm_path"]]
        volume = np.stack([dicom_to_hu(dataset) for dataset in datasets])
        positions = ordered["image_position_z"].to_numpy(float)
        source_instances = ordered["instance_number"].astype(int).tolist()
        for thickness in thicknesses:
            slabs, centers, contributors = reconstruct_thick_volume(
                volume, positions, thickness=thickness
            )
            series_root = (
                output_root
                / "png_thick"
                / f"{thickness:g}mm"
                / f"{split}_images"
                / study_uid
            )
            for slab_index, (slab, center, indices) in enumerate(
                zip(slabs, centers, contributors), start=1
            ):
                png_path = series_root / f"{slab_index}.png"
                image = hu_to_uint8(slab)
                if png_path.exists() and not overwrite:
                    decoded = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
                    if decoded is None or decoded.shape != image.shape:
                        raise ValueError(f"Incompatible existing PNG: {png_path}")
                else:
                    _write_png_atomic(png_path, image)
                records.append(
                    {
                        "split": split,
                        "study_uid": study_uid,
                        "target_thickness": float(thickness),
                        "target_interval": float(thickness),
                        "slab_index": slab_index - 1,
                        "center_z": float(center),
                        "png_path": str(png_path.resolve()),
                        "source_instances": json.dumps(
                            [source_instances[index] for index in indices]
                        ),
                    }
                )
    result = pd.DataFrame(records)
    manifest_path = output_root / "manifests" / f"{split}_thick_slices.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(manifest_path, index=False)
    return result
