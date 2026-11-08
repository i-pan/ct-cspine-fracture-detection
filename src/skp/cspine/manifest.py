from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from skp.cspine.constants import LEVELS, TARGET_COLUMNS
from skp.cspine.dicom import convert_slice, discover_dicoms
from skp.cspine.folds import assign_multilabel_folds


def _convert_job(job: tuple[str, str, str, bool]) -> dict:
    dcm_path, png_path, split, overwrite = job
    return convert_slice(
        dcm_path, png_path, split=split, overwrite=overwrite
    ).to_dict()


def convert_dicom_tree(
    data_root: str | Path,
    output_root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "test"),
    workers: int = 0,
    overwrite: bool = False,
) -> pd.DataFrame:
    data_root = Path(data_root)
    output_root = Path(output_root)
    jobs: list[tuple[str, str, str, bool]] = []
    for split in splits:
        for dcm_path in discover_dicoms(data_root, split):
            png_path = (
                output_root
                / "png"
                / f"{split}_images"
                / dcm_path.parent.name
                / f"{dcm_path.stem}.png"
            )
            jobs.append((str(dcm_path), str(png_path), split, overwrite))

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_convert_job, jobs, chunksize=32))
    else:
        records = [_convert_job(job) for job in jobs]
    manifest = pd.DataFrame.from_records(records)
    return add_derived_geometry(manifest)


def add_derived_geometry(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.sort_values(
        ["split", "study_uid", "image_position_z", "instance_number"]
    ).reset_index(drop=True)
    manifest["slice_index"] = manifest.groupby(
        ["split", "study_uid"], sort=False
    ).cumcount()
    manifest["actual_z_spacing"] = np.nan
    for _, indices in manifest.groupby(["split", "study_uid"]).groups.items():
        index = np.asarray(list(indices))
        positions = manifest.loc[index, "image_position_z"].to_numpy(float)
        if len(positions) == 1:
            spacing = np.array([np.nan])
        else:
            differences = np.abs(np.diff(positions))
            spacing = np.r_[differences[0], (differences[:-1] + differences[1:]) / 2, differences[-1]]
        manifest.loc[index, "actual_z_spacing"] = spacing
    return manifest


def build_study_table(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    studies = pd.read_csv(data_root / "train.csv")
    boxes = pd.read_csv(data_root / "train_bounding_boxes.csv")
    segmented = {
        path.stem for path in (data_root / "segmentations").glob("*.nii")
    }
    studies["has_boxes"] = studies["StudyInstanceUID"].isin(
        boxes["StudyInstanceUID"]
    ).astype(np.int8)
    studies["has_segmentation"] = studies["StudyInstanceUID"].isin(segmented).astype(
        np.int8
    )
    if studies["patient_overall"].ne(studies.loc[:, LEVELS].max(axis=1)).any():
        raise ValueError("patient_overall is inconsistent with C1-C7 labels.")
    if studies.loc[:, TARGET_COLUMNS].isna().any().any():
        raise ValueError("Training labels contain missing values.")
    return studies


def build_folds(
    data_root: str | Path, *, n_splits: int = 5, seed: int = 2022
) -> pd.DataFrame:
    studies = build_study_table(data_root)
    columns = (*TARGET_COLUMNS, "has_boxes", "has_segmentation")
    return assign_multilabel_folds(
        studies, n_splits=n_splits, seed=seed, columns=columns
    )


def attach_training_metadata(
    manifest: pd.DataFrame,
    studies: pd.DataFrame,
    boxes: pd.DataFrame,
) -> pd.DataFrame:
    result = manifest.copy()
    train_mask = result["split"].eq("train")
    training = result.loc[train_mask].merge(
        studies,
        left_on="study_uid",
        right_on="StudyInstanceUID",
        how="left",
        validate="many_to_one",
    )
    if training["StudyInstanceUID"].isna().any():
        raise ValueError("At least one training DICOM study has no study-level label.")
    result.loc[train_mask, "fold"] = training["fold"].to_numpy()
    for column in TARGET_COLUMNS:
        result.loc[train_mask, column] = training[column].to_numpy()
    result["fold"] = result["fold"].fillna(-1).astype(np.int8)

    grouped_boxes = (
        boxes.assign(
            bbox_xywh=boxes.apply(
                lambda row: [
                    float(row.x),
                    float(row.y),
                    float(row.width),
                    float(row.height),
                ],
                axis=1,
            )
        )
        .groupby(["StudyInstanceUID", "slice_number"])["bbox_xywh"]
        .apply(list)
    )
    keys = list(zip(result["study_uid"], result["instance_number"]))
    result["fracture_boxes"] = [grouped_boxes.get(key, []) for key in keys]
    result["has_fracture_box"] = result["fracture_boxes"].map(bool).astype(np.int8)
    return result


def write_project_manifests(
    data_root: str | Path,
    output_root: str | Path,
    slice_manifest: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 2022,
) -> tuple[Path, Path]:
    output_root = Path(output_root)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    studies = build_folds(data_root, n_splits=n_splits, seed=seed)
    boxes = pd.read_csv(Path(data_root) / "train_bounding_boxes.csv")
    slices = attach_training_metadata(slice_manifest, studies, boxes)
    study_path = manifest_root / "studies.parquet"
    slice_path = manifest_root / "slices.parquet"
    studies.to_parquet(study_path, index=False)
    slices.to_parquet(slice_path, index=False)
    return study_path, slice_path
