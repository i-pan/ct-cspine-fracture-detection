from __future__ import annotations

import numpy as np
import pandas as pd

from skp.cspine.constants import TARGET_COLUMNS


def assign_multilabel_folds(
    studies: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 2022,
    columns: tuple[str, ...] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Greedily balance multilabel prevalence and fold size.

    Studies are assigned once and remain the unit of leakage control. Annotation
    availability columns may be appended to ``columns`` by the caller.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    missing = sorted(set(columns) - set(studies.columns))
    if missing:
        raise ValueError(f"Missing stratification columns: {missing}")
    if studies["StudyInstanceUID"].duplicated().any():
        raise ValueError("StudyInstanceUID must be unique before fold assignment.")

    result = studies.copy().reset_index(drop=True)
    labels = result.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Stratification columns must be binary.")

    rng = np.random.default_rng(seed)
    rarity = 1.0 / np.maximum(labels.sum(axis=0), 1.0)
    priority = labels @ rarity + rng.uniform(0, 1e-6, size=len(result))
    order = np.argsort(-priority)
    desired_labels = labels.sum(axis=0) / n_splits
    desired_size = len(result) / n_splits
    fold_labels = np.zeros((n_splits, labels.shape[1]), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    folds = np.full(len(result), -1, dtype=np.int64)

    for index in order:
        candidate_scores = []
        for fold in range(n_splits):
            label_cost = (
                labels[index]
                * fold_labels[fold]
                / np.maximum(desired_labels, 1.0)
            ).sum()
            size_cost = fold_sizes[fold] / desired_size
            candidate_scores.append(label_cost + 0.5 * size_cost)
        minimum = min(candidate_scores)
        candidates = np.flatnonzero(np.isclose(candidate_scores, minimum))
        fold = int(rng.choice(candidates))
        folds[index] = fold
        fold_labels[fold] += labels[index]
        fold_sizes[fold] += 1

    result["fold"] = folds
    return result


def validate_fold_isolation(
    slice_manifest: pd.DataFrame, study_folds: pd.DataFrame
) -> None:
    mapping = study_folds.set_index("StudyInstanceUID")["fold"]
    observed = slice_manifest.groupby("study_uid")["fold"].nunique()
    if (observed != 1).any():
        raise ValueError("At least one study spans multiple folds.")
    expected = slice_manifest["study_uid"].map(mapping)
    if expected.isna().any() or not np.array_equal(
        expected.to_numpy(), slice_manifest["fold"].to_numpy()
    ):
        raise ValueError("Slice manifest fold values disagree with study folds.")
