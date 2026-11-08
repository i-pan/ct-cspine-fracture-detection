from __future__ import annotations

import numpy as np
import pandas as pd


def build_slice_teacher_manifest(
    predictions: pd.DataFrame,
    slices: pd.DataFrame,
    boxes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach adjacent PNGs and manual evaluation targets to native predictions."""
    required_predictions = {
        "StudyInstanceUID",
        "instance_number",
        "slice_index",
        "logit_max",
    }
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing native prediction columns: {missing}")

    train_slices = slices[slices["split"] == "train"].copy()
    train_slices["StudyInstanceUID"] = train_slices["study_uid"].astype(str)
    train_slices = train_slices.sort_values(["StudyInstanceUID", "slice_index"])
    grouped = train_slices.groupby("StudyInstanceUID", sort=False)["png_path"]
    train_slices["png_path_prev"] = grouped.shift(1).fillna(train_slices["png_path"])
    train_slices["png_path_next"] = grouped.shift(-1).fillna(train_slices["png_path"])

    columns = [
        "StudyInstanceUID",
        "instance_number",
        "slice_index",
        "fold",
        "png_path_prev",
        "png_path",
        "png_path_next",
    ]
    manifest = predictions.merge(
        train_slices[columns],
        on=["StudyInstanceUID", "instance_number", "slice_index"],
        how="inner",
        validate="one_to_one",
    )
    manifest = manifest.rename(columns={"png_path": "png_path_center"})
    manifest["teacher_probability"] = 1.0 / (
        1.0 + np.exp(-manifest["logit_max"].clip(-30, 30))
    )

    box_studies = set(boxes["StudyInstanceUID"].astype(str))
    box_slices = {
        (str(row.StudyInstanceUID), int(row.slice_number))
        for row in boxes.itertuples(index=False)
    }
    manifest["manual_available"] = manifest["StudyInstanceUID"].isin(box_studies)
    manifest["manual_target"] = [
        float((str(study), int(instance)) in box_slices)
        for study, instance in zip(
            manifest["StudyInstanceUID"], manifest["instance_number"]
        )
    ]
    return manifest.sort_values(
        ["StudyInstanceUID", "slice_index"]
    ).reset_index(drop=True)


def contiguous_pseudo_labels(
    probabilities: np.ndarray,
    *,
    bag_target: int,
    positive_threshold: float = 0.7,
    negative_threshold: float = 0.1,
    max_run: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return slice targets and supervision mask for one vertebra bag.

    Ambiguous values are represented by ``mask=False`` rather than fabricated
    negatives.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("probabilities must be a nonempty one-dimensional array.")
    targets = np.zeros_like(probabilities, dtype=np.float32)
    mask = np.zeros_like(probabilities, dtype=bool)
    if not bag_target:
        mask[:] = True
        return targets, mask

    mask[probabilities <= negative_threshold] = True
    peak = int(np.argmax(probabilities))
    if probabilities[peak] < positive_threshold:
        return targets, mask
    accepted = probabilities >= positive_threshold
    left = peak
    right = peak
    while left > 0 and accepted[left - 1] and right - left + 1 < max_run:
        left -= 1
    while (
        right + 1 < len(probabilities)
        and accepted[right + 1]
        and right - left + 1 < max_run
    ):
        right += 1
    targets[left : right + 1] = 1.0
    mask[left : right + 1] = True
    return targets, mask


def generate_pseudolabel_table(
    predictions: pd.DataFrame,
    *,
    probability_columns: list[str],
    positive_threshold: float = 0.7,
    negative_threshold: float = 0.1,
) -> pd.DataFrame:
    required = {"StudyInstanceUID", "level", "target", "teacher_fold"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    rows = []
    for row in predictions.itertuples(index=False):
        probabilities = np.asarray(
            [getattr(row, column) for column in probability_columns], dtype=float
        )
        targets, mask = contiguous_pseudo_labels(
            probabilities,
            bag_target=int(row.target),
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
        for index, (target, supervised, probability) in enumerate(
            zip(targets, mask, probabilities)
        ):
            rows.append(
                {
                    "StudyInstanceUID": row.StudyInstanceUID,
                    "level": row.level,
                    "slice_index": index,
                    "pseudo_target": float(target),
                    "pseudo_supervised": bool(supervised),
                    "teacher_probability": float(probability),
                    "teacher_fold": int(row.teacher_fold),
                }
            )
    return pd.DataFrame(rows)
