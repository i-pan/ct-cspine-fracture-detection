from __future__ import annotations

import json
import os
import subprocess
import tempfile
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom

from skp.cspine.constants import LEVELS
from skp.cspine.crops import resize_volume
from skp.cspine.segmentations import load_rsna_segmentation


TOTALSEGMENTATOR_VERSION = "2.16.0"
TOTALSEGMENTATOR_TASK = "total"
TOTALSEGMENTATOR_RESAMPLING_ORDER = 3


def _missing_levels(value) -> list[str]:
    """Normalize a parquet- or JSON-backed missing-level value."""
    if value is None:
        return []
    if isinstance(value, str):
        value = literal_eval(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [str(level) for level in value]


def build_totalsegmentator_qc_manifest(samples: pd.DataFrame) -> pd.DataFrame:
    """Keep teacher samples containing at least one cervical vertebral level.

    TotalSegmentator can occasionally return an all-background cervical mask.
    Partial cervical coverage is retained because it is expected in the source
    dataset and the segmentation loss handles absent classes.
    """
    required = {"StudyInstanceUID", "missing_levels", "volume_path"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"Teacher manifest is missing columns: {sorted(missing)}")
    keep = samples["missing_levels"].map(
        lambda value: len(set(_missing_levels(value)) & set(LEVELS)) < len(LEVELS)
    )
    result = samples.loc[keep].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("Teacher QC rejected every pseudo-labeled sample.")
    return result


def dicom_series_to_nifti(dicom_paths: list[Path], output_path: Path) -> np.ndarray:
    """Write a temporary HU NIfTI and return physical DICOM slice order."""
    datasets = [pydicom.dcmread(str(path)) for path in dicom_paths]
    if len(datasets) < 2:
        raise ValueError("TotalSegmentator requires at least two DICOM slices.")
    orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
    row_direction, column_direction = orientation[:3], orientation[3:]
    normal = np.cross(row_direction, column_direction)
    positions = np.asarray(
        [dataset.ImagePositionPatient for dataset in datasets], dtype=float
    )
    order = np.argsort(positions @ normal)
    datasets = [datasets[index] for index in order]
    positions = positions[order]
    volume = np.stack(
        [
            dataset.pixel_array.astype(np.float32)
            * float(getattr(dataset, "RescaleSlope", 1.0))
            + float(getattr(dataset, "RescaleIntercept", 0.0))
            for dataset in datasets
        ]
    )
    spacing = np.asarray(datasets[0].PixelSpacing, dtype=float)
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = row_direction * spacing[1]
    affine_lps[:3, 1] = column_direction * spacing[0]
    affine_lps[:3, 2] = np.mean(np.diff(positions, axis=0), axis=0)
    affine_lps[:3, 3] = positions[0]
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    image = nib.Nifti1Image(
        np.rint(volume).astype(np.int16).transpose(2, 1, 0),
        lps_to_ras @ affine_lps,
    )
    nib.save(image, output_path)
    return order


def get_totalsegmentator_level_ids(python_path: Path) -> dict[int, int]:
    """Read C1-C7 IDs from the installed TotalSegmentator class map."""
    code = (
        "import json; "
        "from totalsegmentator.map_to_binary import class_map; "
        "print(json.dumps(class_map['total']))"
    )
    result = subprocess.run(
        [str(python_path), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    class_map = json.loads(result.stdout)
    name_to_id = {name: int(index) for index, name in class_map.items()}
    return {
        name_to_id[f"vertebrae_{level}"]: index
        for index, level in enumerate(LEVELS, start=1)
    }


def verify_totalsegmentator_version(python_path: Path) -> None:
    code = (
        "from importlib.metadata import version; "
        "print(version('TotalSegmentator'))"
    )
    result = subprocess.run(
        [str(python_path), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    installed = result.stdout.strip()
    if installed != TOTALSEGMENTATOR_VERSION:
        raise RuntimeError(
            "TotalSegmentator version mismatch: "
            f"expected {TOTALSEGMENTATOR_VERSION}, found {installed}."
        )


def remap_totalsegmentator_mask(
    mask: np.ndarray, level_ids: dict[int, int]
) -> np.ndarray:
    remapped = np.zeros(mask.shape, dtype=np.uint8)
    for source, target in level_ids.items():
        remapped[mask == source] = target
    return remapped


def dice_by_level(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} versus {target.shape}")
    scores = {}
    for index, level in enumerate(LEVELS, start=1):
        predicted = prediction == index
        expected = target == index
        denominator = predicted.sum() + expected.sum()
        scores[f"dice_{level}"] = (
            float(2 * np.logical_and(predicted, expected).sum() / denominator)
            if denominator
            else 1.0
        )
    scores["dice_mean"] = float(
        np.mean([scores[f"dice_{level}"] for level in LEVELS])
    )
    return scores


def _load_png_volume(rows: pd.DataFrame, physical_order: np.ndarray) -> np.ndarray:
    images = [
        cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in rows["png_path"]
    ]
    if any(image is None for image in images):
        raise FileNotFoundError("Could not decode a PNG while building teacher cache.")
    return np.stack(images)[physical_order]


def _run_teacher(
    *,
    study_uid: str,
    rows: pd.DataFrame,
    data_root: Path,
    output_root: Path,
    executable: Path,
    level_ids: dict[int, int],
    device: int,
    save_volume: bool,
) -> dict:
    mode = "pseudo" if save_volume else "benchmark"
    root = output_root / "totalsegmentator_teacher"
    record_root = root / f"{mode}_records"
    volume_root = root / "volumes"
    record_root.mkdir(parents=True, exist_ok=True)
    if save_volume:
        volume_root.mkdir(parents=True, exist_ok=True)
    record_path = record_root / f"{study_uid}.json"
    volume_path = volume_root / f"{study_uid}.npz"
    if record_path.is_file() and (not save_volume or volume_path.is_file()):
        return json.loads(record_path.read_text())

    ordered = rows.sort_values("instance_number").reset_index(drop=True)
    dicom_paths = [Path(path) for path in ordered["dcm_path"]]
    temporary_root = output_root / "teacher_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{study_uid}-", dir=temporary_root
    ) as temporary:
        temporary = Path(temporary)
        ct_path = temporary / "ct.nii.gz"
        segmentation_path = temporary / "segmentation.nii.gz"
        report_path = temporary / "report.json"
        log_path = temporary / "totalsegmentator.log"
        physical_order = dicom_series_to_nifti(dicom_paths, ct_path)
        command = [
            str(executable),
            "-i",
            str(ct_path),
            "-o",
            str(segmentation_path),
            "-ml",
            "-ta",
            TOTALSEGMENTATOR_TASK,
            "-ro",
            str(TOTALSEGMENTATOR_RESAMPLING_ORDER),
            "-d",
            f"gpu:{device}",
            "-rp",
            str(report_path),
        ]
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(
                f"TotalSegmentator failed for {study_uid} on GPU {device}:\n{tail}"
            )
        raw = np.asarray(nib.load(segmentation_path).get_fdata(), dtype=np.uint8)
        native_mask = remap_totalsegmentator_mask(
            raw.transpose(2, 1, 0), level_ids
        )
        if native_mask.shape[0] != len(ordered):
            raise ValueError(
                f"{study_uid}: teacher depth {native_mask.shape[0]} does not "
                f"match DICOM depth {len(ordered)}"
            )
        missing_levels = [
            level
            for index, level in enumerate(LEVELS, start=1)
            if not np.any(native_mask == index)
        ]
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        result = {
            "StudyInstanceUID": study_uid,
            "fold": int(ordered["fold"].iloc[0]),
            "device": device,
            "teacher_version": TOTALSEGMENTATOR_VERSION,
            "teacher_task": TOTALSEGMENTATOR_TASK,
            "resampling_order": TOTALSEGMENTATOR_RESAMPLING_ORDER,
            "source_depth": int(native_mask.shape[0]),
            "source_height": int(native_mask.shape[1]),
            "source_width": int(native_mask.shape[2]),
            "missing_levels": missing_levels,
            "teacher_report_json": json.dumps(report, sort_keys=True),
        }
        if save_volume:
            image = _load_png_volume(ordered, physical_order)
            resized_image = resize_volume(
                image, (128, 128, 128), mode="trilinear"
            )
            resized_mask = resize_volume(
                native_mask, (128, 128, 128), mode="nearest"
            )
            temporary_volume = volume_path.with_suffix(".npz.tmp")
            with temporary_volume.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    image=np.rint(resized_image).astype(np.uint8),
                    mask=np.rint(resized_mask).astype(np.uint8),
                )
            os.replace(temporary_volume, volume_path)
            result["volume_path"] = str(volume_path.resolve())
        else:
            positions = ordered["image_position_z"].to_numpy(float)
            manual = load_rsna_segmentation(
                data_root / "segmentations" / f"{study_uid}.nii", positions
            )
            manual = manual[np.argsort(positions)]
            result.update(dice_by_level(native_mask, manual))

    temporary_record = record_path.with_suffix(".json.tmp")
    temporary_record.write_text(json.dumps(result, sort_keys=True))
    os.replace(temporary_record, record_path)
    return result


def run_totalsegmentator_teacher(
    *,
    slices: pd.DataFrame,
    studies: pd.DataFrame,
    data_root: str | Path,
    output_root: str | Path,
    environment: str | Path,
    devices: list[int],
    manual_only: bool,
) -> pd.DataFrame:
    if not devices:
        raise ValueError("At least one GPU device is required.")
    data_root = Path(data_root)
    output_root = Path(output_root)
    environment = Path(environment)
    executable = environment / "bin" / "TotalSegmentator"
    python_path = environment / "bin" / "python"
    if not executable.is_file() or not python_path.is_file():
        raise FileNotFoundError(f"Invalid TotalSegmentator environment: {environment}")
    verify_totalsegmentator_version(python_path)
    level_ids = get_totalsegmentator_level_ids(python_path)
    eligible = studies[
        studies["has_segmentation"].eq(1 if manual_only else 0)
    ]
    uids = sorted(eligible["StudyInstanceUID"].astype(str))
    partitions = [uids[index :: len(devices)] for index in range(len(devices))]

    def process_partition(device: int, partition: list[str]) -> list[dict]:
        records = []
        for study_uid in partition:
            rows = slices[
                slices["split"].eq("train")
                & slices["study_uid"].eq(study_uid)
            ]
            records.append(
                _run_teacher(
                    study_uid=study_uid,
                    rows=rows,
                    data_root=data_root,
                    output_root=output_root,
                    executable=executable,
                    level_ids=level_ids,
                    device=device,
                    save_volume=not manual_only,
                )
            )
            print(
                f"GPU {device}: completed {len(records)}/{len(partition)} "
                f"({study_uid})",
                flush=True,
            )
        return records

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(process_partition, device, partition)
            for device, partition in zip(devices, partitions)
        ]
        records = [record for future in futures for record in future.result()]
    result = pd.DataFrame(records).sort_values("StudyInstanceUID")
    manifest_name = (
        "totalsegmentator_benchmark.parquet"
        if manual_only
        else "totalsegmentator_samples.parquet"
    )
    result.to_parquet(output_root / "manifests" / manifest_name, index=False)
    if not manual_only:
        qc = build_totalsegmentator_qc_manifest(result)
        qc.to_parquet(
            output_root / "manifests" / "totalsegmentator_samples_qc.parquet",
            index=False,
        )
    return result
