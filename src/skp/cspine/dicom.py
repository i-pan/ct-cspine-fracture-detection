from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from skp.cspine.constants import WINDOW_LOWER, WINDOW_UPPER


def _require_pydicom():
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DICOM conversion requires the medical dependency group: "
            "`uv sync --group medical`."
        ) from exc
    return pydicom


@dataclass(frozen=True)
class SliceRecord:
    split: str
    study_uid: str
    series_uid: str
    sop_uid: str
    patient_id: str
    instance_number: int
    dcm_path: str
    png_path: str
    rows: int
    columns: int
    image_position_x: float
    image_position_y: float
    image_position_z: float
    orientation: tuple[float, ...]
    pixel_spacing_y: float
    pixel_spacing_x: float
    slice_thickness: float
    spacing_between_slices: float
    rescale_slope: float
    rescale_intercept: float
    photometric_interpretation: str
    bits_stored: int
    pixel_representation: int
    png_sha256: str

    def to_dict(self) -> dict:
        record = asdict(self)
        record["orientation"] = list(self.orientation)
        return record


def hu_to_uint8(
    image_hu: np.ndarray,
    lower: float = WINDOW_LOWER,
    upper: float = WINDOW_UPPER,
) -> np.ndarray:
    """Apply a deterministic fixed CT window and quantize to uint8."""
    if not upper > lower:
        raise ValueError(f"upper must exceed lower, got {lower=} and {upper=}")
    image = np.asarray(image_hu, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("HU image contains non-finite values.")
    image = np.clip(image, lower, upper)
    image = np.rint((image - lower) * (255.0 / (upper - lower)))
    return image.astype(np.uint8)


def dicom_to_hu(dataset) -> np.ndarray:
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = dataset.pixel_array.astype(np.float32)
    return pixels * slope + intercept


def _float_values(value, expected: int, default: float = np.nan) -> tuple[float, ...]:
    if value is None:
        return (default,) * expected
    values = tuple(float(item) for item in value)
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {values}")
    return values


def read_geometry(path: str | Path):
    pydicom = _require_pydicom()
    return pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=False,
    )


def _write_png_atomic(path: Path, image: np.ndarray, compression: int = 3) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not success:
        raise RuntimeError(f"OpenCV failed to encode {path}")
    payload = encoded.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return digest


def convert_slice(
    dcm_path: str | Path,
    png_path: str | Path,
    *,
    split: str,
    overwrite: bool = False,
) -> SliceRecord:
    pydicom = _require_pydicom()
    dcm_path = Path(dcm_path)
    png_path = Path(png_path)
    dataset = pydicom.dcmread(str(dcm_path), force=False)
    photo = str(getattr(dataset, "PhotometricInterpretation", ""))
    if photo != "MONOCHROME2":
        raise ValueError(
            f"Unsupported PhotometricInterpretation={photo!r} in {dcm_path}"
        )

    image = hu_to_uint8(dicom_to_hu(dataset))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM, got shape {image.shape} in {dcm_path}")

    if png_path.exists() and not overwrite:
        existing = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if existing is None or existing.shape != image.shape or existing.dtype != np.uint8:
            raise ValueError(f"Existing PNG is incompatible with {dcm_path}: {png_path}")
        digest = hashlib.sha256(png_path.read_bytes()).hexdigest()
    else:
        digest = _write_png_atomic(png_path, image)

    position = _float_values(getattr(dataset, "ImagePositionPatient", None), 3)
    orientation = _float_values(
        getattr(dataset, "ImageOrientationPatient", None), 6
    )
    spacing = _float_values(getattr(dataset, "PixelSpacing", None), 2)
    instance = int(getattr(dataset, "InstanceNumber"))
    if dcm_path.stem.isdigit() and instance != int(dcm_path.stem):
        raise ValueError(
            f"InstanceNumber {instance} does not match filename {dcm_path.name}"
        )

    return SliceRecord(
        split=split,
        study_uid=str(dataset.StudyInstanceUID),
        series_uid=str(dataset.SeriesInstanceUID),
        sop_uid=str(dataset.SOPInstanceUID),
        patient_id=str(dataset.PatientID),
        instance_number=instance,
        dcm_path=str(dcm_path.resolve()),
        png_path=str(png_path.resolve()),
        rows=int(dataset.Rows),
        columns=int(dataset.Columns),
        image_position_x=position[0],
        image_position_y=position[1],
        image_position_z=position[2],
        orientation=orientation,
        pixel_spacing_y=spacing[0],
        pixel_spacing_x=spacing[1],
        slice_thickness=float(getattr(dataset, "SliceThickness", np.nan)),
        spacing_between_slices=float(
            getattr(dataset, "SpacingBetweenSlices", np.nan)
        ),
        rescale_slope=float(getattr(dataset, "RescaleSlope", 1.0)),
        rescale_intercept=float(getattr(dataset, "RescaleIntercept", 0.0)),
        photometric_interpretation=photo,
        bits_stored=int(getattr(dataset, "BitsStored", 0)),
        pixel_representation=int(getattr(dataset, "PixelRepresentation", 0)),
        png_sha256=digest,
    )


def discover_dicoms(data_root: str | Path, split: str) -> Iterable[Path]:
    image_root = Path(data_root) / f"{split}_images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing DICOM directory: {image_root}")
    return sorted(
        image_root.glob("*/*.dcm"),
        key=lambda path: (path.parent.name, int(path.stem)),
    )
