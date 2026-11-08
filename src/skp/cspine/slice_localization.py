from __future__ import annotations

import ast
import copy
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


def resampled_source_instances(
    mask: np.ndarray,
    source_instances,
    *,
    target_depth: int,
    mask_threshold: float = 0.4,
    stack_size: int | None = None,
) -> list[int]:
    """Map close-square model positions back to native DICOM instances."""
    instances = np.asarray(_parse_sequence(source_instances), dtype=np.int64)
    if mask.ndim != 3 or len(mask) != len(instances):
        raise ValueError("Mask depth and source_instances must agree.")
    threshold = float(mask_threshold)
    coordinates = np.argwhere(mask >= threshold)
    while coordinates.size == 0 and threshold > 0.1:
        threshold = round(threshold - 0.1, 1)
        coordinates = np.argwhere(mask >= threshold)
    if coordinates.size == 0:
        raise ValueError("Mask has no voxels at the configured thresholds.")

    lower = int(coordinates[:, 0].min())
    upper = int(coordinates[:, 0].max()) + 1
    cropped = instances[lower:upper]
    source_depth = len(cropped)
    if source_depth > target_depth:
        # PyTorch linear interpolation with align_corners=False samples these
        # source-bin centers. Associate each center with its nearest instance.
        coordinates_1d = (
            (np.arange(target_depth, dtype=np.float64) + 0.5)
            * source_depth
            / target_depth
            - 0.5
        )
        indices = np.floor(coordinates_1d + 0.5).astype(np.int64)
        indices = np.clip(indices, 0, source_depth - 1)
    elif source_depth < target_depth:
        # This is PyTorch's nearest-neighbor upsampling index rule.
        indices = np.floor(
            np.arange(target_depth, dtype=np.float64)
            * source_depth
            / target_depth
        ).astype(np.int64)
    else:
        indices = np.arange(target_depth, dtype=np.int64)
    mapped = cropped[indices]

    if stack_size is not None:
        if stack_size <= 0 or stack_size % 2 == 0:
            raise ValueError("stack_size must be a positive odd integer.")
        if target_depth % stack_size:
            raise ValueError("target_depth must be divisible by stack_size.")
        mapped = mapped.reshape(-1, stack_size)[:, stack_size // 2]
    return mapped.astype(int).tolist()


def _parse_sequence(value) -> list:
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def interpolate_crop_logits(
    logits: np.ndarray,
    source_instances,
    native_slices: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate resampled crop logits back to native DICOM slice indices."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    instances = [int(value) for value in _parse_sequence(source_instances)]
    if len(logits) != len(instances):
        raise ValueError(
            f"Got {len(logits)} logits for {len(instances)} source instances."
        )
    instance_to_index = dict(
        zip(
            native_slices["instance_number"].astype(int),
            native_slices["slice_index"].astype(int),
        )
    )
    pairs = [
        (instance_to_index[instance], logit)
        for instance, logit in zip(instances, logits)
        if instance in instance_to_index
    ]
    if not pairs:
        return pd.DataFrame(columns=["instance_number", "slice_index", "logit"])

    # Shallow upsampled crops can map several model positions to one native slice.
    grouped: dict[int, list[float]] = defaultdict(list)
    for index, logit in pairs:
        grouped[index].append(float(logit))
    source_indices = np.array(sorted(grouped), dtype=np.int64)
    source_logits = np.array(
        [np.mean(grouped[index]) for index in source_indices], dtype=np.float64
    )
    lower, upper = int(source_indices.min()), int(source_indices.max())
    selected = native_slices[
        native_slices["slice_index"].between(lower, upper)
    ].copy()
    selected["logit"] = np.interp(
        selected["slice_index"].to_numpy(), source_indices, source_logits
    )
    return selected[["instance_number", "slice_index", "logit"]]


def map_crop_predictions_to_native_levels(
    crop_predictions: pd.DataFrame,
    slices: pd.DataFrame,
) -> pd.DataFrame:
    """Map crop logits to the native grid while retaining vertebral level."""
    records: list[pd.DataFrame] = []
    slices_by_study = {
        study: frame.sort_values("slice_index")
        for study, frame in slices.groupby("study_uid", sort=False)
    }
    for row in crop_predictions.itertuples(index=False):
        native = slices_by_study.get(row.StudyInstanceUID)
        if native is None:
            continue
        mapped = interpolate_crop_logits(row.slice_logits, row.source_instances, native)
        if mapped.empty:
            continue
        mapped["StudyInstanceUID"] = row.StudyInstanceUID
        mapped["level"] = row.level
        records.append(mapped)
    if not records:
        raise ValueError("No crop predictions could be mapped to native slices.")

    return pd.concat(records, ignore_index=True)


def aggregate_mapped_level_predictions(mapped: pd.DataFrame) -> pd.DataFrame:
    """Aggregate native-grid C1-C7 logits into one score per CT slice."""

    keys = ["StudyInstanceUID", "instance_number", "slice_index"]
    return (
        mapped.groupby(keys, as_index=False)
        .agg(
            logit_max=("logit", "max"),
            logit_mean=("logit", "mean"),
            n_levels=("logit", "size"),
        )
        .sort_values(["StudyInstanceUID", "slice_index"])
        .reset_index(drop=True)
    )


def aggregate_native_predictions(
    crop_predictions: pd.DataFrame,
    slices: pd.DataFrame,
) -> pd.DataFrame:
    """Map crop logits to native slices and aggregate overlapping C1-C7 crops."""
    mapped = map_crop_predictions_to_native_levels(crop_predictions, slices)
    return aggregate_mapped_level_predictions(mapped)


def ensemble_native_predictions(
    first_crops: pd.DataFrame,
    second_crops: pd.DataFrame,
    slices: pd.DataFrame,
) -> pd.DataFrame:
    """Average two models per native slice and level before level pooling."""
    first = map_crop_predictions_to_native_levels(first_crops, slices)
    second = map_crop_predictions_to_native_levels(second_crops, slices)
    keys = ["StudyInstanceUID", "instance_number", "slice_index", "level"]
    merged = first.merge(second, on=keys, how="outer", suffixes=("_first", "_second"))
    merged["logit"] = merged[["logit_first", "logit_second"]].mean(axis=1)
    return aggregate_mapped_level_predictions(merged[keys + ["logit"]])


def evaluate_native_predictions(
    predictions: pd.DataFrame,
    boxes: pd.DataFrame,
) -> dict[str, float | int]:
    annotated = {
        (str(row.StudyInstanceUID), int(row.slice_number))
        for row in boxes.itertuples(index=False)
    }
    evaluated = predictions.copy()
    evaluated["target"] = [
        int((str(study), int(instance)) in annotated)
        for study, instance in zip(
            evaluated["StudyInstanceUID"], evaluated["instance_number"]
        )
    ]
    metrics: dict[str, float | int] = {
        "studies": int(evaluated["StudyInstanceUID"].nunique()),
        "slices": int(len(evaluated)),
        "positive_slices": int(evaluated["target"].sum()),
    }
    target = evaluated["target"].to_numpy()
    for aggregation in ("max", "mean"):
        scores = evaluated[f"logit_{aggregation}"].to_numpy()
        metrics[f"auc_{aggregation}"] = float(roc_auc_score(target, scores))
        metrics[f"ap_{aggregation}"] = float(average_precision_score(target, scores))
        study_aucs = []
        for _, frame in evaluated.groupby("StudyInstanceUID"):
            if frame["target"].nunique() == 2:
                study_aucs.append(
                    roc_auc_score(frame["target"], frame[f"logit_{aggregation}"])
                )
        metrics[f"study_auc_{aggregation}"] = float(np.mean(study_aucs))

    for k in (1, 3, 5, 10):
        hits = []
        for _, frame in evaluated.groupby("StudyInstanceUID"):
            if not frame["target"].any():
                continue
            hits.append(bool(frame.nlargest(k, "logit_max")["target"].any()))
        metrics[f"top{k}_study_recall"] = float(np.mean(hits))

    box_slices = boxes[["StudyInstanceUID", "slice_number"]].drop_duplicates()
    box_keys = {
        (str(study), int(instance))
        for study, instance in box_slices.itertuples(index=False, name=None)
    }
    prediction_keys = {
        (str(study), int(instance))
        for study, instance in evaluated[
            ["StudyInstanceUID", "instance_number"]
        ].itertuples(index=False, name=None)
    }
    metrics["annotated_slices"] = len(box_keys)
    metrics["covered_annotated_slices"] = len(box_keys & prediction_keys)
    return metrics


@torch.inference_mode()
def infer_crop_slice_logits(
    *,
    checkpoint: Path,
    config_name: str,
    manifest_path: Path,
    study_ids: set[str],
    fold: int,
    device: str,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    config_module = import_module(f"skp.configs.{config_name}")
    cfg = copy.deepcopy(config_module.cfg)
    cfg.pretrained = False
    # Activation checkpointing only adds recomputation overhead under inference.
    cfg.enable_gradient_checkpointing = False
    # Training uses small chunks to bound activation memory. Inference retains no
    # activations, so exploit the available VRAM to reduce encoder launches.
    if cfg.get("encoder_chunk_size"):
        cfg.encoder_chunk_size = 128
    cfg.fold = fold
    cfg.annotations_file = str(manifest_path)
    cfg.val_batch_size = batch_size
    cfg.num_workers = num_workers

    dataset_module = import_module(f"skp.datasets.{cfg.dataset}")
    dataset = dataset_module.Dataset(cfg, "val")
    dataset.df = dataset.df[
        dataset.df["StudyInstanceUID"].astype(str).isin(study_ids)
    ].reset_index(drop=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=dataset.collate_fn,
    )

    model_module = import_module(f"skp.models.{cfg.model}")
    model = model_module.Net(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(state)
    model.to(device).eval()

    outputs = []
    use_amp = device.startswith("cuda")
    for batch in loader:
        indices = batch["index"].numpy()
        model_batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "index"
        }
        with torch.autocast(
            device_type="cuda" if use_amp else "cpu",
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = model(model_batch)
        logits = output["slice_logits"].squeeze(-1).float().cpu().numpy()
        for index, values in zip(indices, logits):
            row = dataset.df.iloc[int(index)]
            source_instances = row.source_instances
            close_square_shape = cfg.get("close_square_mil_output_shape")
            if close_square_shape:
                with np.load(row.crop_path) as data:
                    source_mask = data["mask"]
                source_instances = resampled_source_instances(
                    source_mask,
                    row.source_instances,
                    target_depth=int(close_square_shape[0]),
                    mask_threshold=float(
                        cfg.get("close_square_mil_mask_threshold", 0.4)
                    ),
                    stack_size=(
                        int(cfg.spaced_stack_size)
                        if cfg.get("spaced_stack_size")
                        else None
                    ),
                )
            if len(values) != len(source_instances):
                raise ValueError(
                    "Slice-logit/native-position mismatch for "
                    f"{row.StudyInstanceUID} {row.level}: "
                    f"{len(values)} != {len(source_instances)}."
                )
            outputs.append(
                {
                    "StudyInstanceUID": str(row.StudyInstanceUID),
                    "level": row.level,
                    "source_instances": source_instances,
                    "slice_logits": values.tolist(),
                }
            )
    return pd.DataFrame(outputs)
