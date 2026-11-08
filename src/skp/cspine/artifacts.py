from __future__ import annotations

import copy
import hashlib
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import torch


ARTIFACT_FORMAT_VERSION = 1


def resolve_pretrained_path(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve a local model directory or download a Hub snapshot."""
    local = Path(model_id_or_path).expanduser()
    if local.is_dir():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading a Hugging Face model ID requires `huggingface-hub`."
        ) from exc
    return Path(
        snapshot_download(
            repo_id=str(model_id_or_path),
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )


def read_artifact_manifest(model_root: Path, expected_type: str) -> dict[str, Any]:
    path = model_root / "cspine_model.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing model manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported artifact format {manifest.get('format_version')}; "
            f"expected {ARTIFACT_FORMAT_VERSION}."
        )
    if manifest.get("model_type") != expected_type:
        raise ValueError(
            f"Expected model_type={expected_type!r}, got "
            f"{manifest.get('model_type')!r}."
        )
    return manifest


def inference_config(config_name: str, fold: int):
    cfg = copy.deepcopy(import_module(f"skp.configs.{config_name}").cfg)
    cfg.fold = int(fold)
    cfg.pretrained = False
    for name in (
        "load_pretrained_model",
        "load_pretrained_backbone",
        "load_pretrained_encoder",
        "load_pretrained_decoder",
    ):
        if hasattr(cfg, name):
            setattr(cfg, name, None)
    if hasattr(cfg, "mil_checkpoint_by_fold"):
        cfg.mil_checkpoint_by_fold = {}
    if hasattr(cfg, "enable_gradient_checkpointing"):
        cfg.enable_gradient_checkpointing = False
    return cfg


def load_artifact_model(
    model_root: Path,
    entry: dict[str, Any],
    *,
    device: str | torch.device,
):
    """Construct an SKP model and strictly load a SafeTensors state dict."""
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Safe model loading requires the `safetensors` package."
        ) from exc
    cfg = inference_config(entry["config"], int(entry["fold"]))
    model = import_module(f"skp.models.{cfg.model}").Net(cfg)
    weights_path = model_root / entry["weights"]
    expected_sha256 = entry.get("sha256")
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(weights_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Weight checksum mismatch for {weights_path}: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), cfg


def checkpoint_model_state(
    checkpoint: str | Path,
    *,
    use_ema: bool,
) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if use_ema:
        state = payload.get("callbacks", {}).get("EMACallback", {}).get(
            "ema_state_dict"
        )
        if state is None:
            raise KeyError(f"Checkpoint has no EMA state: {checkpoint}")
    else:
        state = payload.get("state_dict", payload)
    result = {
        key.removeprefix("model."): value.detach().cpu().contiguous()
        for key, value in state.items()
        if key.startswith("model.") and not key.startswith("model.criterion.")
    }
    if not result:
        raise ValueError(f"No model weights found in {checkpoint}")
    return result


def save_safetensors_atomic(
    state: dict[str, torch.Tensor], output_path: Path
) -> None:
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(state, str(temporary))
    temporary.replace(output_path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
