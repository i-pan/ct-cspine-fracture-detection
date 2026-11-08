from __future__ import annotations

import argparse
import copy
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skp.environment import load_project_dotenv
from skp.cspine.release_manifest import (
    DEFAULT_RELEASE_MANIFEST,
    best_checkpoint,
    component_spec,
    load_release_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"


def checkpoint_for_family(
    family: str,
    fold: int,
    root: Path | None = None,
    *,
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT,
    release_manifest: Path = DEFAULT_RELEASE_MANIFEST,
) -> Path:
    if root is not None:
        path = root / f"fold{fold}/checkpoints/best.ckpt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    manifest = load_release_manifest(release_manifest)
    specification = component_spec(manifest, f"{family}.mil")
    return best_checkpoint(experiment_root, specification, fold=fold)


def load_ema_mil_model(
    config_name: str,
    checkpoint: Path,
    *,
    fold: int,
    device: str,
):
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.fold = fold
    cfg.pretrained = False
    # Activation checkpointing is useful for training but only recomputes work
    # and adds overhead under inference_mode.
    cfg.enable_gradient_checkpointing = False
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema = payload.get("callbacks", {}).get("EMACallback", {}).get("ema_state_dict")
    if ema is None:
        raise KeyError(f"Checkpoint has no EMACallback state: {checkpoint}")
    state = {
        key.removeprefix("model."): value
        for key, value in ema.items()
        if key.startswith("model.") and not key.startswith("model.criterion.")
    }
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


def assemble_series_cache(
    studies: pd.DataFrame,
    vertebrae: pd.DataFrame,
    row_features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Align per-crop features into an explicit C1-C7 series tensor."""
    if len(vertebrae) != len(row_features):
        raise ValueError("Every vertebra row must have exactly one feature vector.")
    if vertebrae.duplicated(["StudyInstanceUID", "level_index"]).any():
        raise ValueError("Vertebra manifest contains duplicate study/level rows.")
    if not vertebrae["level_index"].between(0, 6).all():
        raise ValueError("level_index must be in [0,6].")
    study_to_index = {
        uid: index for index, uid in enumerate(studies["StudyInstanceUID"])
    }
    features = np.zeros(
        (len(studies), 7, row_features.shape[1]), dtype=row_features.dtype
    )
    valid_mask = np.zeros((len(studies), 7), dtype=bool)
    for row_index, row in vertebrae.reset_index(drop=True).iterrows():
        study_index = study_to_index.get(row.StudyInstanceUID)
        if study_index is None:
            raise ValueError(f"Unknown study in vertebra manifest: {row.StudyInstanceUID}")
        level_index = int(row.level_index)
        features[study_index, level_index] = row_features[row_index]
        valid_mask[study_index, level_index] = True
    target_columns = ["patient_overall"] + [f"C{level}" for level in range(1, 8)]
    return {
        "features": features,
        "valid_mask": valid_mask,
        "targets": studies[target_columns].to_numpy(dtype=np.uint8),
        "fold": studies["fold"].to_numpy(dtype=np.int8),
        "study_uid": studies["StudyInstanceUID"].to_numpy(dtype="U64"),
    }


def assemble_series_depth_cache(
    studies: pd.DataFrame,
    vertebrae: pd.DataFrame,
    row_features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Align per-depth crop features into an explicit C1-C7 tensor."""
    if row_features.ndim != 3:
        raise ValueError(
            f"Expected per-crop features shaped (N,D,F), got {row_features.shape}."
        )
    if len(vertebrae) != len(row_features):
        raise ValueError("Every vertebra row must have exactly one feature sequence.")
    if vertebrae.duplicated(["StudyInstanceUID", "level_index"]).any():
        raise ValueError("Vertebra manifest contains duplicate study/level rows.")
    if not vertebrae["level_index"].between(0, 6).all():
        raise ValueError("level_index must be in [0,6].")
    study_to_index = {
        uid: index for index, uid in enumerate(studies["StudyInstanceUID"])
    }
    depth, feature_dim = row_features.shape[1:]
    features = np.zeros(
        (len(studies), 7, depth, feature_dim), dtype=row_features.dtype
    )
    valid_mask = np.zeros((len(studies), 7, depth), dtype=bool)
    for row_index, row in vertebrae.reset_index(drop=True).iterrows():
        study_index = study_to_index.get(row.StudyInstanceUID)
        if study_index is None:
            raise ValueError(f"Unknown study in vertebra manifest: {row.StudyInstanceUID}")
        level_index = int(row.level_index)
        features[study_index, level_index] = row_features[row_index]
        valid_mask[study_index, level_index] = True
    target_columns = ["patient_overall"] + [f"C{level}" for level in range(1, 8)]
    return {
        "features": features,
        "valid_mask": valid_mask,
        "targets": studies[target_columns].to_numpy(dtype=np.uint8),
        "fold": studies["fold"].to_numpy(dtype=np.int8),
        "study_uid": studies["StudyInstanceUID"].to_numpy(dtype="U64"),
    }


@torch.inference_mode()
def extract_fold(
    *,
    family: str,
    fold: int,
    output_path: Path,
    studies_path: Path,
    checkpoint_root: Path | None,
    experiment_root: Path,
    release_manifest: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    encoder_chunk_size: int,
    representation: str,
) -> None:
    manifest = load_release_manifest(release_manifest)
    config_name = component_spec(manifest, f"{family}.mil")["config"]
    checkpoint = checkpoint_for_family(
        family,
        fold,
        checkpoint_root,
        experiment_root=experiment_root,
        release_manifest=release_manifest,
    )
    model, cfg = load_ema_mil_model(
        config_name,
        checkpoint,
        fold=fold,
        device=device,
    )
    model.encoder_chunk_size = encoder_chunk_size
    dataset = import_module(f"skp.datasets.{cfg.dataset}").Dataset(cfg, "inference")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate_fn,
    )
    if representation == "pooled":
        # Pooled BiGRU feature (512), bag logit, MIL logit, normalized
        # attention entropy, and maximum attention.
        row_features = np.empty((len(dataset), 516), dtype=np.float16)
    elif representation == "depth":
        # Contextual BiGRU feature (512), slice logit, and attention weight at
        # each of the 32 depth positions. No metadata features are included.
        row_features = np.empty((len(dataset), 32, 514), dtype=np.float16)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    for batch_index, batch in enumerate(loader):
        indices = batch["index"].numpy()
        model_batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if torch.is_tensor(value) and key != "index"
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            out = model(model_batch, return_features=True)
        attention = out["attention"].float()
        if representation == "pooled":
            entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
            entropy = entropy / np.log(attention.shape[1])
            auxiliary = torch.stack(
                [
                    out["logits"].flatten().float(),
                    out["mil_logits"].flatten().float(),
                    entropy,
                    attention.amax(dim=1),
                ],
                dim=1,
            )
            combined = torch.cat([out["features"].float(), auxiliary], dim=1)
        else:
            combined = torch.cat(
                [
                    out["features_seq"].float(),
                    out["slice_logits"].float(),
                    attention.unsqueeze(-1),
                ],
                dim=-1,
            )
            if combined.shape[1:] != (32, 514):
                raise ValueError(
                    "Depth representation must have shape (B,32,514), got "
                    f"{tuple(combined.shape)}."
                )
        row_features[indices] = combined.cpu().numpy().astype(np.float16)
        if batch_index % 100 == 0:
            print(
                f"family={family} fold={fold} "
                f"rows={min((batch_index + 1) * batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    studies = pd.read_parquet(studies_path)
    if representation == "pooled":
        cache = assemble_series_cache(studies, dataset.df, row_features)
    else:
        cache = assemble_series_depth_cache(studies, dataset.df, row_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete archive. Downstream fold queues watch for the
    # final filename and must never observe an in-progress ZIP central directory.
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **cache)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(output_path)
    missing = int((~cache["valid_mask"]).sum())
    print(
        f"saved={output_path} shape={cache['features'].shape} missing_levels={missing}",
        flush=True,
    )


def main() -> None:
    load_project_dotenv()
    work_root = Path(os.environ.get("CSPINE_WORK_ROOT", "data/processed"))
    release = load_release_manifest(DEFAULT_RELEASE_MANIFEST)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=sorted(release["selected_pipeline"]["families"]),
        required=True,
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=work_root / "series_features",
    )
    parser.add_argument(
        "--studies",
        type=Path,
        default=work_root / "manifests" / "studies.parquet",
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument(
        "--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--encoder-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--representation", choices=["pooled", "depth"], default="pooled"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.representation == "depth" and args.output_root == work_root / "series_features":
        args.output_root = args.output_root.with_name("series_features_7x32")
    for fold in [int(value) for value in args.folds.split(",")]:
        output_path = args.output_root / args.family / f"fold{fold}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}", flush=True)
            continue
        extract_fold(
            family=args.family,
            fold=fold,
            output_path=output_path,
            studies_path=args.studies,
            checkpoint_root=args.checkpoint_root,
            experiment_root=args.experiment_root,
            release_manifest=args.release_manifest,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            encoder_chunk_size=args.encoder_chunk_size,
            representation=args.representation,
        )


if __name__ == "__main__":
    main()
