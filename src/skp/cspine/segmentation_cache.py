from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from skp.cspine.crops import resize_volume
from skp.cspine.segmentations import load_rsna_segmentation


def build_segmentation_cache(
    slices: pd.DataFrame,
    studies: pd.DataFrame,
    segmentation_root: str | Path,
    output_root: str | Path,
    *,
    output_shape: tuple[int, int, int] = (128, 128, 128),
) -> pd.DataFrame:
    segmentation_root = Path(segmentation_root)
    output_root = Path(output_root)
    cache_root = output_root / "segmentation_volumes"
    cache_root.mkdir(parents=True, exist_ok=True)
    studies_by_uid = studies.set_index("StudyInstanceUID")
    rows = []
    for segmentation_path in sorted(segmentation_root.glob("*.nii")):
        study_uid = segmentation_path.stem
        group = slices[
            slices["split"].eq("train") & slices["study_uid"].eq(study_uid)
        ].sort_values("instance_number")
        if group.empty:
            raise ValueError(f"No PNG manifest rows found for {study_uid}.")
        images = [
            cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in group["png_path"]
        ]
        if any(image is None for image in images):
            raise FileNotFoundError(f"Could not decode a PNG for {study_uid}.")
        volume = np.stack(images)
        mask = load_rsna_segmentation(
            segmentation_path, group["image_position_z"].to_numpy(float)
        )
        physical_order = np.argsort(group["image_position_z"].to_numpy(float))
        volume = volume[physical_order]
        mask = mask[physical_order]
        image_resized = resize_volume(volume, output_shape, mode="trilinear")
        mask_resized = resize_volume(mask, output_shape, mode="nearest")
        cache_path = cache_root / f"{study_uid}.npz"
        np.savez_compressed(
            cache_path,
            image=np.rint(image_resized).astype(np.uint8),
            mask=np.rint(mask_resized).astype(np.uint8),
        )
        study = studies_by_uid.loc[study_uid]
        rows.append(
            {
                "StudyInstanceUID": study_uid,
                "volume_path": str(cache_path.resolve()),
                "fold": int(study["fold"]),
                "source_depth": int(volume.shape[0]),
                "source_height": int(volume.shape[1]),
                "source_width": int(volume.shape[2]),
            }
        )
    annotations = pd.DataFrame(rows)
    path = output_root / "manifests" / "segmentation_samples.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations.to_parquet(path, index=False)
    return annotations
