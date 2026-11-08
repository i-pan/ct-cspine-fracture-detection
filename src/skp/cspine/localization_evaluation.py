from __future__ import annotations

import argparse
import copy
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from skp.cspine.slice_localization import (
    aggregate_native_predictions,
    evaluate_native_predictions,
    resampled_source_instances,
)


def _load_ema_localizer(config_name: str, checkpoint: Path, fold: int, device: str):
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.fold = fold
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema = payload.get("callbacks", {}).get("EMACallback", {}).get("ema_state_dict")
    if ema is None:
        raise KeyError(f"Checkpoint has no EMACallback state: {checkpoint}")
    state = {
        key.removeprefix("model."): value
        for key, value in ema.items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


def _loader(config_name: str, fold: int, batch_size: int, num_workers: int):
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.fold = fold
    dataset = import_module(f"skp.datasets.{cfg.dataset}").Dataset(cfg, "val")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    return dataset, loader


def _spatial_accumulator() -> dict[str, list]:
    return {
        "slice_target": [],
        "slice_score": [],
        "pointing": [],
        "dice": [],
    }


def _update_spatial_metrics(
    accumulator: dict[str, list],
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> None:
    slice_target = target.flatten(2).any(dim=2)
    slice_score = prediction.flatten(2).amax(dim=2)
    accumulator["slice_target"].extend(
        slice_target.flatten().cpu().numpy().astype(int)
    )
    accumulator["slice_score"].extend(slice_score.flatten().cpu().numpy())
    positive = slice_target
    if not positive.any():
        return
    positive_target = target[positive].flatten(1).bool()
    positive_prediction = prediction[positive].flatten(1)
    peak = positive_prediction.argmax(dim=1)
    pointing = positive_target[
        torch.arange(len(peak), device=peak.device), peak
    ].float()
    binary = positive_prediction >= threshold
    intersection = (binary & positive_target).sum(dim=1).float()
    denominator = binary.sum(dim=1) + positive_target.sum(dim=1)
    dice = 2 * intersection / denominator.clamp_min(1)
    accumulator["pointing"].extend(pointing.cpu().numpy())
    accumulator["dice"].extend(dice.cpu().numpy())


def _compute_spatial_metrics(accumulator: dict[str, list]) -> dict[str, float]:
    target = np.asarray(accumulator["slice_target"])
    score = np.asarray(accumulator["slice_score"])
    return {
        "common_grid_auc": float(roc_auc_score(target, score)),
        "common_grid_ap": float(average_precision_score(target, score)),
        "pointing_accuracy": float(np.mean(accumulator["pointing"])),
        "positive_dice": float(np.mean(accumulator["dice"])),
        "positions": int(len(target)),
        "positive_positions": int(target.sum()),
    }


def _checkpoint(root: Path, fold: int) -> Path:
    path = root / f"fold{fold}" / "checkpoints" / "best.ckpt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


@torch.inference_mode()
def evaluate_localization_ensemble(
    *,
    d32_root: Path,
    stack3_root: Path,
    slices_path: Path,
    boxes_path: Path,
    output_root: Path,
    folds: list[int],
    batch_size: int = 4,
    num_workers: int = 4,
    d32_device: str = "cuda:0",
    stack3_device: str = "cuda:1",
    threshold: float = 0.3,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    slices = pd.read_parquet(slices_path)
    boxes = pd.read_csv(boxes_path)
    source_manifest = pd.read_parquet(
        slices_path.parent / "vertebra_samples_native_depth.parquet"
    )
    source_instances_by_crop = source_manifest.set_index("crop_path")[
        "source_instances"
    ].to_dict()
    all_native: dict[str, list[pd.DataFrame]] = {
        "d32": [],
        "stack3": [],
        "ensemble": [],
    }
    fold_reports = []
    for fold in folds:
        d32_dataset, d32_loader = _loader(
            "cspine_fracture_localization_d32", fold, batch_size, num_workers
        )
        stack3_dataset, stack3_loader = _loader(
            "cspine_fracture_localization_3x32", fold, batch_size, num_workers
        )
        keys = ["StudyInstanceUID", "level", "crop_path"]
        if not d32_dataset.df[keys].equals(stack3_dataset.df[keys]):
            raise ValueError(f"Localization manifests are not aligned in fold {fold}.")
        d32_model, _ = _load_ema_localizer(
            "cspine_fracture_localization_d32",
            _checkpoint(d32_root, fold),
            fold,
            d32_device,
        )
        stack3_model, _ = _load_ema_localizer(
            "cspine_fracture_localization_3x32",
            _checkpoint(stack3_root, fold),
            fold,
            stack3_device,
        )
        accumulators = {
            name: _spatial_accumulator()
            for name in ("d32", "stack3", "ensemble")
        }
        crop_records = {name: [] for name in accumulators}
        for d32_batch, stack3_batch in zip(d32_loader, stack3_loader, strict=True):
            if not torch.equal(d32_batch["index"], stack3_batch["index"]):
                raise ValueError("D32 and 3x32 loader indices diverged.")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d32_logits = d32_model({"x": d32_batch["x"].to(d32_device)})[
                    "logits"
                ]
                stack3_logits = stack3_model(
                    {"x": stack3_batch["x"].to(stack3_device)}
                )["logits"]
            d32_probability = F.interpolate(
                d32_logits.float().sigmoid(),
                size=(96, 96, 96),
                mode="trilinear",
                align_corners=False,
            )[:, 0].cpu()
            stack3_probability = (
                stack3_logits.float()
                .sigmoid()
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, 96, 96, 96)
                .cpu()
            )
            target = (
                stack3_batch["y"]
                .permute(0, 2, 1, 3, 4)
                .reshape(-1, 96, 96, 96)
                .bool()
            )
            predictions = {
                "d32": d32_probability,
                "stack3": stack3_probability,
                "ensemble": (d32_probability + stack3_probability) / 2,
            }
            indices = d32_batch["index"].tolist()
            for name, prediction in predictions.items():
                _update_spatial_metrics(
                    accumulators[name], prediction, target, threshold
                )
                scores = prediction.flatten(2).amax(dim=2).numpy()
                for index, values in zip(indices, scores):
                    row = d32_dataset.df.iloc[index]
                    with np.load(row.crop_path) as data:
                        source_mask = data["mask"]
                    source_instances = resampled_source_instances(
                        source_mask,
                        source_instances_by_crop[row.crop_path],
                        target_depth=96,
                        mask_threshold=0.4,
                    )
                    crop_records[name].append(
                        {
                            "StudyInstanceUID": str(row.StudyInstanceUID),
                            "level": row.level,
                            "source_instances": source_instances,
                            "slice_logits": values.tolist(),
                        }
                    )

        fold_boxes = boxes[
            boxes["StudyInstanceUID"].astype(str).isin(
                d32_dataset.df["StudyInstanceUID"].astype(str)
            )
        ]
        fold_report = {"fold": fold, "models": {}}
        for name in accumulators:
            native = aggregate_native_predictions(
                pd.DataFrame(crop_records[name]), slices
            )
            native["fold"] = fold
            native.to_parquet(
                output_root / f"fold{fold}_{name}_native.parquet", index=False
            )
            all_native[name].append(native)
            fold_report["models"][name] = {
                **_compute_spatial_metrics(accumulators[name]),
                **{
                    f"native_{key}": value
                    for key, value in evaluate_native_predictions(
                        native, fold_boxes
                    ).items()
                },
            }
        fold_reports.append(fold_report)
        del d32_model, stack3_model
        torch.cuda.empty_cache()

    report = {"folds": fold_reports, "summary": {}}
    for name, frames in all_native.items():
        pooled = pd.concat(frames, ignore_index=True)
        pooled.to_parquet(output_root / f"oof_{name}_native.parquet", index=False)
        model_fold_metrics = [entry["models"][name] for entry in fold_reports]
        numeric_keys = model_fold_metrics[0].keys()
        mean_fold = {
            key: float(np.mean([metrics[key] for metrics in model_fold_metrics]))
            for key in numeric_keys
        }
        report["summary"][name] = {
            "mean_fold": mean_fold,
            "pooled_native": evaluate_native_predictions(
                pooled,
                boxes[
                    boxes["StudyInstanceUID"].astype(str).isin(
                        pooled["StudyInstanceUID"].astype(str)
                    )
                ],
            ),
        }
    (output_root / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d32-root", type=Path, required=True)
    parser.add_argument("--stack3-root", type=Path, required=True)
    parser.add_argument("--slices", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    report = evaluate_localization_ensemble(
        d32_root=args.d32_root,
        stack3_root=args.stack3_root,
        slices_path=args.slices,
        boxes_path=args.boxes,
        output_root=args.output_root,
        folds=[int(item) for item in args.folds.split(",")],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
