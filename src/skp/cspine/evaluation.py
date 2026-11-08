from __future__ import annotations

import numpy as np


def weighted_binary_log_loss(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    positive_weights: np.ndarray | None = None,
    negative_weights: np.ndarray | None = None,
    eps: float = 1e-7,
) -> float:
    """RSNA-style weighted log loss for columns overall,C1,...,C7."""
    targets = np.asarray(targets, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("targets and probabilities must have equal 2D shapes.")
    classes = targets.shape[1]
    if positive_weights is None:
        positive_weights = np.array([14.0] + [2.0] * (classes - 1))
    if negative_weights is None:
        negative_weights = np.array([7.0] + [1.0] * (classes - 1))
    positive_weights = np.asarray(positive_weights, dtype=float)
    negative_weights = np.asarray(negative_weights, dtype=float)
    if positive_weights.shape != (classes,) or negative_weights.shape != (classes,):
        raise ValueError("Weight vectors must have one entry per target column.")
    probabilities = np.clip(probabilities, eps, 1 - eps)
    weights = np.where(targets == 1, positive_weights, negative_weights)
    loss = -weights * (
        targets * np.log(probabilities)
        + (1 - targets) * np.log(1 - probabilities)
    )
    return float(loss.sum() / weights.sum())


def topk_slice_recall(
    probabilities: np.ndarray,
    annotated: np.ndarray,
    *,
    k: int,
    tolerance: int = 0,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    annotated = np.asarray(annotated, dtype=bool)
    if probabilities.shape != annotated.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and annotated must have shape (N,D).")
    hits = []
    for scores, truth in zip(probabilities, annotated):
        true_indices = np.flatnonzero(truth)
        if not len(true_indices):
            continue
        predicted = np.argsort(-scores)[:k]
        hits.append(
            any(
                np.min(np.abs(true_indices - prediction)) <= tolerance
                for prediction in predicted
            )
        )
    return float(np.mean(hits)) if hits else float("nan")
