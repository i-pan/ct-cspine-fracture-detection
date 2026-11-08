from __future__ import annotations

import argparse
import copy
import json
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from skp.cspine.artifacts import checkpoint_model_state
from skp.cspine.evaluation import weighted_binary_log_loss
from skp.cspine.release_manifest import (
    DEFAULT_RELEASE_MANIFEST,
    best_checkpoint,
    component_spec,
    load_release_manifest,
)


TARGET_NAMES = ["overall"] + [f"c{level}" for level in range(1, 8)]


def probability_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("Targets and probabilities must have equal 2D shapes.")
    if targets.shape[1] != len(TARGET_NAMES):
        raise ValueError("Series evaluation requires overall plus C1-C7 columns.")
    metrics = {
        "competition_log_loss": weighted_binary_log_loss(targets, probabilities)
    }
    level_aucs = []
    for index, name in enumerate(TARGET_NAMES):
        if np.unique(targets[:, index]).size != 2:
            raise ValueError(f"Target {name} does not contain both classes.")
        auc = float(roc_auc_score(targets[:, index], probabilities[:, index]))
        metrics[f"auc_{name}"] = auc
        if index:
            level_aucs.append(auc)
    metrics["auc_levels_mean"] = float(np.mean(level_aucs))
    metrics["auc_mean"] = float(
        np.mean([metrics["auc_overall"], metrics["auc_levels_mean"]])
    )
    return metrics


def align_oof_members(
    members: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Align independently inferred members by study UID and verify targets."""
    if not members:
        raise ValueError("At least one OOF member is required.")
    reference_name = next(iter(members))
    reference_uids, reference_targets, _ = members[reference_name]
    order = np.argsort(reference_uids.astype(str))
    uids = reference_uids[order].astype(str)
    if len(np.unique(uids)) != len(uids):
        raise ValueError(f"Duplicate study UIDs in {reference_name}.")
    targets = reference_targets[order]
    aligned = {}
    for name, (member_uids, member_targets, member_probabilities) in members.items():
        member_order = np.argsort(member_uids.astype(str))
        sorted_uids = member_uids[member_order].astype(str)
        if not np.array_equal(sorted_uids, uids):
            raise ValueError(f"Study UID mismatch for OOF member {name}.")
        sorted_targets = member_targets[member_order]
        if not np.array_equal(sorted_targets, targets):
            raise ValueError(f"Target mismatch for OOF member {name}.")
        aligned[name] = member_probabilities[member_order]
    return uids, targets, aligned


def _feature_cache(
    reference: str,
    fold: int,
    pooled_feature_root: Path,
    depth_feature_root: Path,
) -> Path:
    family, component = reference.split(".", maxsplit=1)
    root = pooled_feature_root if component == "series_7x1" else depth_feature_root
    path = root / family / f"fold{fold}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


@torch.inference_mode()
def infer_member_oof(
    *,
    reference: str,
    specification: dict,
    series_experiment_root: Path,
    pooled_feature_root: Path,
    depth_feature_root: Path,
    folds: list[int],
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_uids = []
    all_targets = []
    all_probabilities = []
    use_ema = specification.get("checkpoint_state") == "ema"
    for fold in folds:
        cfg = copy.deepcopy(import_module(f"skp.configs.{specification['config']}").cfg)
        cfg.fold = fold
        cfg.feature_cache_template = str(
            _feature_cache(
                reference,
                fold,
                pooled_feature_root,
                depth_feature_root,
            )
        )
        model = import_module(f"skp.models.{cfg.model}").Net(cfg)
        checkpoint = best_checkpoint(
            series_experiment_root,
            specification,
            fold=fold,
        )
        model.load_state_dict(
            checkpoint_model_state(checkpoint, use_ema=use_ema), strict=True
        )
        model = model.to(device).eval()
        dataset = import_module(f"skp.datasets.{cfg.dataset}").Dataset(cfg, "val")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=device.startswith("cuda"),
            collate_fn=dataset.collate_fn,
        )
        for batch in loader:
            model_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if torch.is_tensor(value) and key not in {"index", "y"}
            }
            logits = model(model_batch)["logits"]
            all_probabilities.append(logits.sigmoid().float().cpu().numpy())
            all_targets.append(batch["y"].float().numpy())
            all_uids.extend(batch["study_uid"])
        print(
            f"member={reference} fold={fold} studies={len(dataset)} checkpoint={checkpoint}",
            flush=True,
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return (
        np.asarray(all_uids, dtype="U64"),
        np.concatenate(all_targets).astype(np.float32),
        np.concatenate(all_probabilities).astype(np.float32),
    )


def evaluate_release_oof(
    *,
    release_manifest: Path,
    series_experiment_root: Path,
    pooled_feature_root: Path,
    depth_feature_root: Path,
    output_root: Path,
    folds: list[int],
    batch_size: int,
    num_workers: int,
    device: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    predictions_path = output_root / "predictions.npz"
    metrics_path = output_root / "metrics.json"
    if (predictions_path.exists() or metrics_path.exists()) and not overwrite:
        raise FileExistsError(
            f"OOF output already exists under {output_root}; pass --overwrite to rebuild."
        )
    manifest = load_release_manifest(release_manifest)
    references = manifest["selected_pipeline"]["series_ensemble"]["members"]
    output_root.mkdir(parents=True, exist_ok=True)
    member_root = output_root / "members"
    member_root.mkdir(parents=True, exist_ok=True)
    members = {}
    for reference in references:
        member_path = member_root / f"{reference.replace('.', '_')}.npz"
        if member_path.is_file() and not overwrite:
            with np.load(member_path, allow_pickle=False) as cached:
                members[reference] = (
                    cached["study_uid"],
                    cached["targets"],
                    cached["probabilities"],
                )
            print(f"member={reference} resume={member_path}", flush=True)
            continue
        member = infer_member_oof(
            reference=reference,
            specification=component_spec(manifest, reference),
            series_experiment_root=series_experiment_root,
            pooled_feature_root=pooled_feature_root,
            depth_feature_root=depth_feature_root,
            folds=folds,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        temporary_member = member_path.with_suffix(".npz.tmp")
        with temporary_member.open("wb") as handle:
            np.savez_compressed(
                handle,
                study_uid=member[0],
                targets=member[1],
                probabilities=member[2],
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_member.replace(member_path)
        members[reference] = member
    uids, targets, aligned = align_oof_members(members)
    ensemble = np.mean(np.stack(list(aligned.values())), axis=0).astype(np.float32)
    probabilities = {**aligned, "ensemble": ensemble}
    metrics = {
        name: probability_metrics(targets, values)
        for name, values in probabilities.items()
    }
    temporary_predictions = predictions_path.with_suffix(".npz.tmp")
    with temporary_predictions.open("wb") as handle:
        np.savez_compressed(
            handle,
            study_uid=uids,
            targets=targets,
            **{
                f"probability_{name.replace('.', '_')}": values
                for name, values in probabilities.items()
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary_predictions.replace(predictions_path)
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    temporary_metrics.replace(metrics_path)
    return predictions_path, metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce OOF predictions and metrics for the selected series ensemble."
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST
    )
    parser.add_argument("--series-experiment-root", type=Path, required=True)
    parser.add_argument("--pooled-feature-root", type=Path, required=True)
    parser.add_argument("--depth-feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    predictions, metrics = evaluate_release_oof(
        release_manifest=args.release_manifest,
        series_experiment_root=args.series_experiment_root,
        pooled_feature_root=args.pooled_feature_root,
        depth_feature_root=args.depth_feature_root,
        output_root=args.output_root,
        folds=[int(fold) for fold in args.folds.split(",")],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(f"predictions={predictions}")
    print(f"metrics={metrics}")


if __name__ == "__main__":
    main()
