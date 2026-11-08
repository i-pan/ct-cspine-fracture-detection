from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NIfTI loading requires the medical dependency group."
        ) from exc
    return nib


def load_rsna_segmentation(
    path: str | Path, image_positions_z: np.ndarray
) -> np.ndarray:
    """Load an RSNA NIfTI mask as a DICOM-ordered ``(D,H,W)`` array.

    This mirrors the orientation correction used by the public third-place
    solution and validates the resulting depth against the DICOM series.
    """
    nib = _require_nibabel()
    raw = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.uint8)
    positions = np.asarray(image_positions_z, dtype=float)
    if len(positions) < 2:
        raise ValueError("A segmentation series must contain at least two slices.")
    z_direction = np.mean(np.diff(positions))
    if z_direction < 0:
        mask = raw[:, ::-1, ::-1].transpose(2, 1, 0)
    else:
        mask = raw[:, ::-1].transpose(2, 1, 0)
    mask[mask > 7] = 0
    if mask.shape[0] != len(positions):
        raise ValueError(
            f"NIfTI depth {mask.shape[0]} does not match DICOM depth {len(positions)}"
        )
    return mask
