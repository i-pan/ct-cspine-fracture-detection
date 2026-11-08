from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from skp.cspine.crops import load_png_volume, resize_volume


def keep_largest_components(mask: np.ndarray, num_levels: int = 7) -> np.ndarray:
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
    for level in range(1, num_levels + 1):
        components, count = ndimage.label(mask == level, structure=structure)
        if not count:
            continue
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        cleaned[components == sizes.argmax()] = level
    return cleaned


def topology_anomaly_levels(
    mask: np.ndarray,
    *,
    num_levels: int = 7,
    max_axial_second_difference: float = 16.0,
) -> list[int]:
    """Find missing or laterally discontinuous vertebral predictions."""
    centroids: list[np.ndarray | None] = []
    anomalous = set()
    for level in range(1, num_levels + 1):
        coordinates = np.argwhere(mask == level)
        if not len(coordinates):
            centroids.append(None)
            anomalous.add(level)
        else:
            centroids.append(coordinates.mean(axis=0))
    for index in range(1, num_levels - 1):
        previous, current, following = centroids[index - 1 : index + 2]
        if previous is None or current is None or following is None:
            continue
        expected_axial = 0.5 * (previous[1:] + following[1:])
        residual = np.linalg.norm(current[1:] - expected_axial)
        if residual > max_axial_second_difference:
            anomalous.add(index + 1)
    return sorted(anomalous)


def _load_model(config_name: str, checkpoint: str | Path, device: torch.device):
    config_module = import_module(f"skp.configs.{config_name}")
    cfg = config_module.cfg.__deepcopy__()
    cfg.pretrained = False
    model_module = import_module(f"skp.models.{cfg.model}")
    model = model_module.Net(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if any(key.startswith("model.") for key in state):
        state = {key: value for key, value in state.items() if key.startswith("model.")}
    state = {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint/model mismatch for {checkpoint}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model.eval().to(device), cfg


@torch.inference_mode()
def infer_segmentation_masks(
    slices: pd.DataFrame,
    studies: pd.DataFrame,
    checkpoints: dict[int, str | Path],
    output_root: str | Path,
    *,
    config_name: str = "cspine_segmentation_x3d_s",
    device: str = "cuda",
    split: str = "train",
    ensemble: bool = False,
    ensemble_on_qc_failure: bool = False,
    study_folds: set[int] | None = None,
    native_grid: bool = True,
) -> Path:
    """Export native-grid masks for crop generation.

    Training studies default to their held-out fold model. Test studies use an
    ensemble when ``ensemble=True``.
    """
    torch_device = torch.device(device)
    models = {
        fold: _load_model(config_name, path, torch_device)
        for fold, path in checkpoints.items()
    }
    mask_root = Path(output_root) / "vertebra_masks" / (
        "ensemble" if ensemble else "oof"
    )
    mask_root.mkdir(parents=True, exist_ok=True)
    qc_root = mask_root / "_qc"
    qc_root.mkdir(exist_ok=True)
    fold_by_study = studies.set_index("StudyInstanceUID")["fold"].to_dict()
    selected = slices[slices["split"].eq(split)]
    if study_folds is not None:
        selected_uids = {
            uid for uid, fold in fold_by_study.items() if int(fold) in study_folds
        }
        selected = selected[selected["study_uid"].isin(selected_uids)]
    for study_uid, group in selected.groupby("study_uid", sort=False):
        native_volume = load_png_volume(group)
        if ensemble:
            selected_models = list(models.values())
        else:
            fold = int(fold_by_study[study_uid])
            if fold not in models:
                raise KeyError(f"No checkpoint supplied for held-out fold {fold}.")
            selected_models = [models[fold]]
        def predict(model_items):
            probabilities = None
            for model, cfg in model_items:
                shape = (cfg.num_slices, cfg.image_height, cfg.image_width)
                volume = resize_volume(native_volume, shape, mode="trilinear")
                tensor = torch.from_numpy(volume[None, None]).float().to(torch_device)
                logits = model({"x": tensor})["logits"]
                prediction = logits.softmax(dim=1)[0].cpu().numpy()
                probabilities = (
                    prediction
                    if probabilities is None
                    else probabilities + prediction
                )
            return probabilities / len(model_items)

        probabilities = predict(selected_models)
        mask = keep_largest_components(
            probabilities.argmax(axis=0).astype(np.uint8)
        )
        initial_anomalies = topology_anomaly_levels(mask)
        used_ensemble_fallback = False
        if initial_anomalies and ensemble_on_qc_failure and not ensemble:
            probabilities = predict(list(models.values()))
            mask = keep_largest_components(
                probabilities.argmax(axis=0).astype(np.uint8)
            )
            used_ensemble_fallback = True
        final_anomalies = topology_anomaly_levels(mask)
        output_mask = (
            resize_volume(mask, native_volume.shape, mode="nearest")
            if native_grid
            else mask
        )
        np.save(mask_root / f"{study_uid}.npy", output_mask.astype(np.uint8))
        (qc_root / f"{study_uid}.json").write_text(
            json.dumps(
                {
                    "StudyInstanceUID": study_uid,
                    "initial_anomaly_levels": initial_anomalies,
                    "final_anomaly_levels": final_anomalies,
                    "used_ensemble_fallback": used_ensemble_fallback,
                },
                sort_keys=True,
            )
        )
    return mask_root
