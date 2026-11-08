from __future__ import annotations

import argparse
import copy
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from skp.cspine.artifacts import checkpoint_model_state
from skp.cspine.release_manifest import (
    DEFAULT_RELEASE_MANIFEST,
    best_checkpoint,
    load_release_manifest,
)
from skp.environment import load_project_dotenv


def multiclass_dice_by_sample(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    classes: range = range(1, 8),
) -> dict[str, float]:
    """Match the validation metric: mean per-sample Dice, then mean classes."""
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have equal shapes.")
    if predictions.ndim < 2:
        raise ValueError("Segmentation arrays require a batch and spatial dimensions.")
    metrics = {}
    for class_index in classes:
        class_scores = []
        for prediction, target in zip(predictions, targets):
            predicted = prediction == class_index
            expected = target == class_index
            denominator = predicted.sum() + expected.sum()
            if denominator:
                class_scores.append(
                    2.0 * np.logical_and(predicted, expected).sum() / denominator
                )
        metrics[f"dice_c{class_index}"] = (
            float(np.mean(class_scores)) if class_scores else float("nan")
        )
    metrics["dice_mean"] = float(np.nanmean(list(metrics.values())))
    return metrics


@torch.inference_mode()
def evaluate_checkpoint_state(
    *,
    config_name: str,
    checkpoint: Path,
    fold: int,
    state: str,
    device: str,
    num_workers: int,
) -> dict[str, float]:
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.fold = fold
    cfg.pretrained = False
    cfg.load_pretrained_model = None
    cfg.enable_gradient_checkpointing = False
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    model.load_state_dict(
        checkpoint_model_state(checkpoint, use_ema=state == "ema"), strict=True
    )
    model = model.to(device).eval()
    dataset = import_module(f"skp.datasets.{cfg.dataset}").Dataset(cfg, "val")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.startswith("cuda"),
        collate_fn=dataset.collate_fn,
    )
    predictions = []
    targets = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            logits = model({"x": x})["logits"]
        predictions.append(logits.argmax(1).cpu().numpy().astype(np.uint8))
        targets.append(batch["y"].numpy().astype(np.uint8))
    result = multiclass_dice_by_sample(
        np.concatenate(predictions), np.concatenate(targets)
    )
    result["studies"] = len(dataset)
    return result


def main() -> None:
    load_project_dotenv()
    parser = argparse.ArgumentParser(
        description="Compare raw and EMA segmentation checkpoint validation Dice."
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST
    )
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--states", default="raw,ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    manifest = load_release_manifest(args.release_manifest)
    specification = manifest["selected_pipeline"]["segmentation"]
    states = args.states.split(",")
    if not set(states) <= {"raw", "ema"}:
        raise ValueError("States must be a comma-separated subset of raw,ema.")
    results = {}
    folds = [int(value) for value in args.folds.split(",")]
    for fold in folds:
        checkpoint = best_checkpoint(args.experiment_root, specification, fold=fold)
        results[f"fold{fold}"] = {}
        for state in states:
            metrics = evaluate_checkpoint_state(
                config_name=specification["config"],
                checkpoint=checkpoint,
                fold=fold,
                state=state,
                device=args.device,
                num_workers=args.num_workers,
            )
            results[f"fold{fold}"][state] = metrics
            print(
                f"fold={fold} state={state} dice_mean={metrics['dice_mean']:.6f}",
                flush=True,
            )
    for state in states:
        results[f"mean_{state}"] = float(
            np.mean([results[f"fold{fold}"][state]["dice_mean"] for fold in folds])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
