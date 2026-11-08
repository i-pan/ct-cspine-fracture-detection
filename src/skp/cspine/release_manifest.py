from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_MANIFEST = PROJECT_ROOT / "release" / "final-v1.json"


def load_release_manifest(path: str | Path = DEFAULT_RELEASE_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported release schema {manifest.get('schema_version')} in {path}."
        )
    selected = manifest.get("selected_pipeline")
    if not isinstance(selected, dict):
        raise ValueError(f"Release manifest has no selected_pipeline: {path}")
    families = selected.get("families")
    if set(families or {}) != {"d32", "3x32"}:
        raise ValueError("Release manifest must define exactly d32 and 3x32 families.")
    members = selected.get("series_ensemble", {}).get("members")
    if not members or len(members) != len(set(members)):
        raise ValueError("Release ensemble members must be a non-empty unique list.")
    return manifest


def component_spec(manifest: dict[str, Any], reference: str) -> dict[str, Any]:
    try:
        family, component = reference.split(".", maxsplit=1)
        return manifest["selected_pipeline"]["families"][family][component]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unknown release component reference: {reference}") from exc


def component_run_root(
    artifact_root: str | Path,
    specification: dict[str, Any],
    *,
    fold: int,
) -> Path:
    overrides = specification.get("fold_overrides", {})
    relative = overrides.get(str(fold), specification["run_root"])
    return Path(artifact_root) / relative


def best_checkpoint(
    artifact_root: str | Path,
    specification: dict[str, Any],
    *,
    fold: int,
) -> Path:
    path = component_run_root(artifact_root, specification, fold=fold) / (
        f"fold{fold}/checkpoints/best.ckpt"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()
