from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skp.cspine.constants import DEFAULT_DATA_ROOT, TARGET_COLUMNS
from skp.cspine.crops import build_vertebra_crops
from skp.cspine.evaluation import weighted_binary_log_loss
from skp.cspine.localization import build_fracture_localization_manifest
from skp.cspine.manifest import (
    build_folds,
    convert_dicom_tree,
    write_project_manifests,
)
from skp.cspine.pseudolabels import (
    build_slice_teacher_manifest,
    generate_pseudolabel_table,
)
from skp.cspine.segmentation_cache import build_segmentation_cache
from skp.cspine.segmentation_inference import infer_segmentation_masks
from skp.cspine.slice_localization import (
    aggregate_native_predictions,
    ensemble_native_predictions,
    evaluate_native_predictions,
    infer_crop_slice_logits,
)
from skp.cspine.thick import build_thick_series
from skp.cspine.totalsegmentator_teacher import run_totalsegmentator_teacher


def _data_output_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def convert_dicom_main() -> None:
    parser = _data_output_parser("Convert RSNA DICOMs to WL400/WW1800 PNGs.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = convert_dicom_tree(
        args.data_root,
        args.output_root,
        splits=tuple(args.splits),
        workers=args.workers,
        overwrite=args.overwrite,
    )
    path = args.output_root / "manifests" / "slices_raw.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
    print(f"Wrote {len(manifest):,} slices to {path}")


def build_folds_main() -> None:
    parser = _data_output_parser("Build leakage-safe study folds.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    studies = build_folds(args.data_root, n_splits=args.folds, seed=args.seed)
    path = args.output_root / "manifests" / "studies.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    studies.to_parquet(path, index=False)
    print(f"Wrote {len(studies):,} studies to {path}")


def build_manifest_main() -> None:
    parser = _data_output_parser("Attach labels, boxes, and folds to slice metadata.")
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    raw_path = args.raw_manifest or (
        args.output_root / "manifests" / "slices_raw.parquet"
    )
    slices = pd.read_parquet(raw_path)
    paths = write_project_manifests(
        args.data_root,
        args.output_root,
        slices,
        n_splits=args.folds,
        seed=args.seed,
    )
    print(f"Wrote manifests: {paths[0]} and {paths[1]}")


def build_segmentation_cache_main() -> None:
    parser = _data_output_parser("Build fixed 128-cube segmentation samples.")
    parser.add_argument("--shape", nargs=3, type=int, default=[128, 128, 128])
    args = parser.parse_args()
    slices = pd.read_parquet(args.output_root / "manifests" / "slices.parquet")
    studies = pd.read_parquet(args.output_root / "manifests" / "studies.parquet")
    result = build_segmentation_cache(
        slices,
        studies,
        args.data_root / "segmentations",
        args.output_root,
        output_shape=tuple(args.shape),
    )
    print(f"Wrote {len(result):,} segmentation samples.")


def totalsegmentator_teacher_main() -> None:
    parser = _data_output_parser(
        "Run full-resolution TotalSegmentator C1-C7 teacher inference."
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--manual-only", action="store_true")
    args = parser.parse_args()
    slices = pd.read_parquet(args.output_root / "manifests" / "slices.parquet")
    studies = pd.read_parquet(args.output_root / "manifests" / "studies.parquet")
    result = run_totalsegmentator_teacher(
        slices=slices,
        studies=studies,
        data_root=args.data_root,
        output_root=args.output_root,
        environment=args.environment,
        devices=[int(device) for device in args.devices.split(",")],
        manual_only=args.manual_only,
    )
    mode = "benchmark" if args.manual_only else "pseudo-label"
    print(f"Wrote {len(result):,} TotalSegmentator {mode} records.")


def build_crops_main() -> None:
    parser = _data_output_parser("Build fixed per-vertebra classification crops.")
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--shape", nargs=3, type=int, default=[32, 384, 384])
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--study-folds",
        help="Optional comma-separated study folds for sharded crop generation.",
    )
    parser.add_argument("--crop-subdir", default="vertebra_crops")
    parser.add_argument("--manifest-name", default="vertebra_samples.parquet")
    parser.add_argument(
        "--variable-depth",
        action="store_true",
        help="Preserve native depth up to --shape depth; downsample only longer crops.",
    )
    parser.add_argument("--exclude-unresolved-qc", action="store_true")
    parser.add_argument(
        "--qc-teacher-annotations",
        type=Path,
        help="Optional teacher manifest used when ensemble topology QC still fails.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    slices = pd.read_parquet(args.output_root / "manifests" / "slices.parquet")
    studies = pd.read_parquet(args.output_root / "manifests" / "studies.parquet")
    result = build_vertebra_crops(
        slices,
        studies,
        args.mask_root,
        args.output_root,
        output_shape=tuple(args.shape),
        padding_fraction=args.padding,
        allow_missing=args.allow_missing,
        study_folds=(
            {int(fold) for fold in args.study_folds.split(",")}
            if args.study_folds
            else None
        ),
        crop_subdir=args.crop_subdir,
        manifest_name=args.manifest_name,
        variable_depth=args.variable_depth,
        exclude_unresolved_qc=args.exclude_unresolved_qc,
        qc_teacher_annotations=args.qc_teacher_annotations,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(result):,} vertebra samples.")


def build_localization_manifest_main() -> None:
    parser = _data_output_parser(
        "Project RSNA fracture boxes into selected per-vertebra MIL crops."
    )
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument(
        "--vertebra-manifest",
        type=Path,
        help="Defaults to manifests/vertebra_samples_native_depth.parquet.",
    )
    parser.add_argument("--target-depth", type=int, default=32)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--close-mask-threshold", type=float, default=0.4)
    parser.add_argument("--close-padding", type=float, default=0.05)
    parser.add_argument(
        "--manifest-name",
        default="fracture_localization_d32.parquet",
    )
    args = parser.parse_args()
    manifest_root = args.output_root / "manifests"
    vertebra_path = args.vertebra_manifest or (
        manifest_root / "vertebra_samples_native_depth.parquet"
    )
    result = build_fracture_localization_manifest(
        pd.read_parquet(vertebra_path),
        pd.read_parquet(manifest_root / "slices.parquet"),
        pd.read_csv(args.data_root / "train_bounding_boxes.csv"),
        args.mask_root,
        target_depth=args.target_depth,
        crop_padding_fraction=args.padding,
        close_mask_threshold=args.close_mask_threshold,
        close_padding_fraction=args.close_padding,
    )
    output_path = manifest_root / args.manifest_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Wrote {len(result):,} localization samples to {output_path}.")


def build_thick_series_main() -> None:
    parser = _data_output_parser("Build deferred thick-slice stress-test PNGs.")
    parser.add_argument("--thickness", nargs="+", type=float, default=[2, 3, 5])
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    slices = pd.read_parquet(args.output_root / "manifests" / "slices.parquet")
    result = build_thick_series(
        slices,
        args.output_root,
        thicknesses=tuple(args.thickness),
        split=args.split,
    )
    print(f"Wrote {len(result):,} reconstructed slabs.")


def infer_segmenter_main() -> None:
    parser = _data_output_parser("Export fold-safe native-grid vertebra masks.")
    parser.add_argument(
        "--checkpoint-fold",
        action="append",
        required=True,
        help="Fold/checkpoint pair such as 0=/path/to/best.ckpt; repeat per fold.",
    )
    parser.add_argument("--config", default="cspine_segmentation_x3d_s")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--checkpoint-state",
        choices=["raw", "ema"],
        default="raw",
        help="Model state to load from each Lightning checkpoint.",
    )
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--ensemble-on-qc-failure", action="store_true")
    parser.add_argument(
        "--compact-grid",
        action="store_true",
        help="Save the 128-cube mask and map crop bounds to native coordinates later.",
    )
    parser.add_argument(
        "--study-folds",
        help="Optional comma-separated study folds for sharded inference.",
    )
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoint_fold:
        fold, path = item.split("=", 1)
        checkpoints[int(fold)] = Path(path)
    slices = pd.read_parquet(args.output_root / "manifests" / "slices.parquet")
    studies = pd.read_parquet(args.output_root / "manifests" / "studies.parquet")
    path = infer_segmentation_masks(
        slices,
        studies,
        checkpoints,
        args.output_root,
        config_name=args.config,
        device=args.device,
        split=args.split,
        ensemble=args.ensemble,
        ensemble_on_qc_failure=args.ensemble_on_qc_failure,
        study_folds=(
            {int(fold) for fold in args.study_folds.split(",")}
            if args.study_folds
            else None
        ),
        native_grid=not args.compact_grid,
        checkpoint_state=args.checkpoint_state,
    )
    print(f"Wrote masks to {path}")


def generate_pseudolabels_main() -> None:
    parser = argparse.ArgumentParser(description="Generate deferred MIL pseudo-labels.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--positive-threshold", type=float, default=0.7)
    parser.add_argument("--negative-threshold", type=float, default=0.1)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    probability_columns = sorted(
        [column for column in predictions if column.startswith("slice_probability_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    result = generate_pseudolabel_table(
        predictions,
        probability_columns=probability_columns,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )
    result.to_parquet(args.output, index=False)
    print(f"Wrote {len(result):,} pseudo slice labels to {args.output}")


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RSNA study probabilities.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    targets = pd.read_parquet(args.targets)
    merged = targets.merge(
        predictions, on="StudyInstanceUID", suffixes=("_target", "_prediction")
    )
    target_values = np.column_stack(
        [merged[f"{column}_target"] for column in TARGET_COLUMNS]
    )
    prediction_values = np.column_stack(
        [merged[f"{column}_prediction"] for column in TARGET_COLUMNS]
    )
    print(f"weighted_log_loss={weighted_binary_log_loss(target_values, prediction_values):.6f}")


def evaluate_slice_localization_main() -> None:
    parser = _data_output_parser(
        "Evaluate weakly supervised slice logits against fracture boxes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="cspine_vertebra_mil_maxvit_tiny")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest_root = args.output_root / "manifests"
    vertebrae = pd.read_parquet(manifest_root / "vertebra_samples.parquet")
    boxes = pd.read_csv(args.data_root / "train_bounding_boxes.csv")
    fold_studies = set(
        vertebrae.loc[vertebrae["fold"] == args.fold, "StudyInstanceUID"].astype(str)
    )
    boxes = boxes[boxes["StudyInstanceUID"].astype(str).isin(fold_studies)].copy()
    study_ids = set(boxes["StudyInstanceUID"].astype(str))
    if not study_ids:
        raise ValueError(f"Fold {args.fold} has no bounding-box annotated studies.")

    crop_predictions = infer_crop_slice_logits(
        checkpoint=args.checkpoint,
        config_name=args.config,
        manifest_path=manifest_root / "vertebra_samples.parquet",
        study_ids=study_ids,
        fold=args.fold,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    slices = pd.read_parquet(manifest_root / "slices.parquet")
    slices = slices[
        (slices["split"] == "train")
        & slices["study_uid"].astype(str).isin(study_ids)
    ]
    native = aggregate_native_predictions(crop_predictions, slices)
    annotated = {
        (str(row.StudyInstanceUID), int(row.slice_number))
        for row in boxes.itertuples(index=False)
    }
    native["target"] = [
        int((str(study), int(instance)) in annotated)
        for study, instance in zip(native.StudyInstanceUID, native.instance_number)
    ]
    native["probability_max"] = 1 / (1 + np.exp(-native["logit_max"]))
    native["probability_mean"] = 1 / (1 + np.exp(-native["logit_mean"]))

    result_root = args.output_root / "evaluations" / "slice_localization"
    result_root.mkdir(parents=True, exist_ok=True)
    crop_path = result_root / f"fold{args.fold}_crop_logits.parquet"
    native_path = result_root / f"fold{args.fold}_native_predictions.parquet"
    crop_predictions.to_parquet(crop_path, index=False)
    native.to_parquet(native_path, index=False)
    metrics = evaluate_native_predictions(native, boxes)
    metrics_path = result_root / f"fold{args.fold}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Saved crop logits to {crop_path}")
    print(f"Saved native predictions to {native_path}")
    print(f"Saved metrics to {metrics_path}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


def _parse_checkpoint_folds(values: list[str]) -> dict[int, Path]:
    checkpoints = {}
    for value in values:
        fold, path = value.split("=", 1)
        checkpoints[int(fold)] = Path(path)
    return checkpoints


def evaluate_slice_localization_cv_main() -> None:
    parser = _data_output_parser(
        "Evaluate two MIL models and their ensemble on held-out box studies."
    )
    parser.add_argument("--d32-checkpoint-fold", action="append", required=True)
    parser.add_argument("--stack3-checkpoint-fold", action="append", required=True)
    parser.add_argument(
        "--d32-config",
        default="cspine_vertebra_mil_maxvit_tiny_d32_close_square",
    )
    parser.add_argument(
        "--stack3-config",
        default="cspine_vertebra_mil_maxvit_tiny_d32_stack3",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    folds = [int(value) for value in args.folds.split(",")]
    d32_checkpoints = _parse_checkpoint_folds(args.d32_checkpoint_fold)
    stack3_checkpoints = _parse_checkpoint_folds(args.stack3_checkpoint_fold)
    missing_d32 = sorted(set(folds) - set(d32_checkpoints))
    missing_stack3 = sorted(set(folds) - set(stack3_checkpoints))
    if missing_d32 or missing_stack3:
        raise ValueError(
            f"Missing checkpoints: d32={missing_d32}, stack3={missing_stack3}."
        )

    manifest_root = args.output_root / "manifests"
    manifest_path = args.manifest or (
        manifest_root / "vertebra_samples_native_depth.parquet"
    )
    vertebrae = pd.read_parquet(manifest_path)
    boxes = pd.read_csv(args.data_root / "train_bounding_boxes.csv")
    slices = pd.read_parquet(manifest_root / "slices.parquet")
    result_root = args.output_root / "evaluations" / "slice_localization_cv"
    result_root.mkdir(parents=True, exist_ok=True)

    per_fold: dict[str, dict] = {}
    pooled: dict[str, list[pd.DataFrame]] = {
        "d32": [],
        "stack3": [],
        "ensemble": [],
    }
    for fold in folds:
        fold_studies = set(
            vertebrae.loc[
                vertebrae["fold"] == fold, "StudyInstanceUID"
            ].astype(str)
        )
        fold_boxes = boxes[
            boxes["StudyInstanceUID"].astype(str).isin(fold_studies)
        ].copy()
        study_ids = set(fold_boxes["StudyInstanceUID"].astype(str))
        if not study_ids:
            raise ValueError(f"Fold {fold} has no bounding-box annotated studies.")
        fold_slices = slices[
            (slices["split"] == "train")
            & slices["study_uid"].astype(str).isin(study_ids)
        ]

        crop_predictions = {}
        for name, config_name, checkpoint in (
            ("d32", args.d32_config, d32_checkpoints[fold]),
            ("stack3", args.stack3_config, stack3_checkpoints[fold]),
        ):
            crop_predictions[name] = infer_crop_slice_logits(
                checkpoint=checkpoint,
                config_name=config_name,
                manifest_path=manifest_path,
                study_ids=study_ids,
                fold=fold,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )

        native_predictions = {
            "d32": aggregate_native_predictions(
                crop_predictions["d32"], fold_slices
            ),
            "stack3": aggregate_native_predictions(
                crop_predictions["stack3"], fold_slices
            ),
            "ensemble": ensemble_native_predictions(
                crop_predictions["d32"],
                crop_predictions["stack3"],
                fold_slices,
            ),
        }
        fold_result_root = result_root / f"fold{fold}"
        fold_result_root.mkdir(parents=True, exist_ok=True)
        per_fold[str(fold)] = {}
        for name in ("d32", "stack3", "ensemble"):
            native = native_predictions[name]
            native.to_parquet(
                fold_result_root / f"{name}_native_predictions.parquet",
                index=False,
            )
            if name != "ensemble":
                crop_predictions[name].to_parquet(
                    fold_result_root / f"{name}_crop_logits.parquet",
                    index=False,
                )
            metrics = evaluate_native_predictions(native, fold_boxes)
            per_fold[str(fold)][name] = metrics
            pooled[name].append(native)
            print(
                f"fold={fold} model={name} auc_max={metrics['auc_max']:.6f} "
                f"ap_max={metrics['ap_max']:.6f}"
            )

    pooled_metrics = {
        name: evaluate_native_predictions(pd.concat(frames), boxes)
        for name, frames in pooled.items()
    }
    summary = {"per_fold": per_fold, "pooled": pooled_metrics}
    summary_path = result_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved cross-validated summary to {summary_path}")
    for name, metrics in pooled_metrics.items():
        print(
            f"pooled model={name} auc_max={metrics['auc_max']:.6f} "
            f"ap_max={metrics['ap_max']:.6f}"
        )


def generate_slice_pseudolabels_cv_main() -> None:
    parser = _data_output_parser(
        "Generate native-slice OOF pseudolabels from the d32 + 3x32 ensemble."
    )
    parser.add_argument("--d32-checkpoint-fold", action="append", required=True)
    parser.add_argument("--stack3-checkpoint-fold", action="append", required=True)
    parser.add_argument(
        "--d32-config",
        default="cspine_vertebra_mil_maxvit_tiny_d32_close_square",
    )
    parser.add_argument(
        "--stack3-config",
        default="cspine_vertebra_mil_maxvit_tiny_d32_stack3",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    folds = [int(value) for value in args.folds.split(",")]
    d32_checkpoints = _parse_checkpoint_folds(args.d32_checkpoint_fold)
    stack3_checkpoints = _parse_checkpoint_folds(args.stack3_checkpoint_fold)
    missing_d32 = sorted(set(folds) - set(d32_checkpoints))
    missing_stack3 = sorted(set(folds) - set(stack3_checkpoints))
    if missing_d32 or missing_stack3:
        raise ValueError(
            f"Missing checkpoints: d32={missing_d32}, stack3={missing_stack3}."
        )

    manifest_root = args.output_root / "manifests"
    crop_manifest = args.manifest or (
        manifest_root / "vertebra_samples_native_depth.parquet"
    )
    vertebrae = pd.read_parquet(crop_manifest)
    slices = pd.read_parquet(manifest_root / "slices.parquet")
    boxes = pd.read_csv(args.data_root / "train_bounding_boxes.csv")
    result_root = args.output_root / "pseudolabels" / "slice_teacher_oof"
    result_root.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        study_ids = set(
            vertebrae.loc[
                vertebrae["fold"] == fold, "StudyInstanceUID"
            ].astype(str)
        )
        fold_slices = slices[
            (slices["split"] == "train")
            & slices["study_uid"].astype(str).isin(study_ids)
        ]
        d32 = infer_crop_slice_logits(
            checkpoint=d32_checkpoints[fold],
            config_name=args.d32_config,
            manifest_path=crop_manifest,
            study_ids=study_ids,
            fold=fold,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        stack3 = infer_crop_slice_logits(
            checkpoint=stack3_checkpoints[fold],
            config_name=args.stack3_config,
            manifest_path=crop_manifest,
            study_ids=study_ids,
            fold=fold,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        native = ensemble_native_predictions(d32, stack3, fold_slices)
        result = build_slice_teacher_manifest(native, fold_slices, boxes)
        path = result_root / f"fold{fold}.parquet"
        result.to_parquet(path, index=False)
        print(
            f"fold={fold} studies={result['StudyInstanceUID'].nunique()} "
            f"slices={len(result)} path={path}"
        )

    expected = [result_root / f"fold{fold}.parquet" for fold in range(5)]
    if all(path.exists() for path in expected):
        combined = pd.concat([pd.read_parquet(path) for path in expected])
        output_path = manifest_root / "slice_teacher_oof.parquet"
        combined.to_parquet(output_path, index=False)
        print(f"Wrote {len(combined):,} OOF slice pseudolabels to {output_path}")
