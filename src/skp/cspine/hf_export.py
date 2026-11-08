from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from skp.cspine.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    checkpoint_model_state,
    save_safetensors_atomic,
    sha256_file,
)
from skp.cspine.release_manifest import (
    DEFAULT_RELEASE_MANIFEST,
    best_checkpoint,
    load_release_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
SERIES_ROOT = EXPERIMENT_ROOT


def _source_reference(
    path: Path,
    source_roots: tuple[tuple[Path, str], ...] = (),
) -> str:
    path = path.resolve()
    for root, prefix in ((PROJECT_ROOT, "project"),) + source_roots:
        try:
            return f"{prefix}/{path.relative_to(root.resolve())}"
        except ValueError:
            continue
    return path.name


def _export_checkpoint(
    checkpoint: Path,
    output_path: Path,
    *,
    use_ema: bool,
    overwrite: bool,
    source_roots: tuple[tuple[Path, str], ...] = (),
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to reuse existing exported weights: {output_path}. "
            "Pass --overwrite to rebuild them from the selected checkpoint."
        )
    state = checkpoint_model_state(checkpoint, use_ema=use_ema)
    save_safetensors_atomic(state, output_path)
    return {
        "weights": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "source_checkpoint": _source_reference(checkpoint, source_roots),
        "source_state": "ema" if use_ema else "model",
    }


def _relative_entry(
    exported: dict[str, Any],
    output_root: Path,
    *,
    config: str,
    fold: int,
) -> dict[str, Any]:
    return {
        "config": config,
        "fold": fold,
        "weights": str(Path(exported["weights"]).relative_to(output_root)),
        "sha256": exported["sha256"],
        "bytes": exported["bytes"],
    }


def _provenance_entry(exported: dict[str, Any], output_root: Path) -> dict[str, Any]:
    return {
        **exported,
        "weights": str(Path(exported["weights"]).relative_to(output_root)),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _build_wheel(output_root: Path) -> str:
    wheel_root = output_root / "python"
    wheel_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_root)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    wheels = sorted(wheel_root.glob("skp-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one SKP wheel, found {wheels}")
    return str(wheels[0].relative_to(output_root))


def _copy_common_files(output_root: Path, wheel_path: str) -> None:
    shutil.copy2(
        PROJECT_ROOT / "requirements" / "inference.txt",
        output_root / "requirements.txt",
    )
    (output_root / "install_local.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'python -m pip install -r "$MODEL_DIR/requirements.txt"\n'
        f'python -m pip install "$MODEL_DIR/{wheel_path}"\n'
    )
    shutil.copy2(PROJECT_ROOT / "LICENSE", output_root / "LICENSE")


def export_segmenter(
    output_root: Path,
    *,
    overwrite: bool,
    experiment_root: Path = EXPERIMENT_ROOT,
    release_manifest: Path = DEFAULT_RELEASE_MANIFEST,
    source_revision: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    release = load_release_manifest(release_manifest)
    source_revision = source_revision or release["source_revision"]
    specification = release["selected_pipeline"]["segmentation"]
    models = []
    provenance = []
    for fold in specification["folds"]:
        checkpoint = best_checkpoint(experiment_root, specification, fold=fold)
        output_path = output_root / "weights" / f"fold{fold}.safetensors"
        exported = _export_checkpoint(
            checkpoint,
            output_path,
            use_ema=specification["checkpoint_state"] == "ema",
            overwrite=overwrite,
            source_roots=((experiment_root, "project-experiments"),),
        )
        models.append(
            _relative_entry(
                exported,
                output_root,
                config=specification["config"],
                fold=fold,
            )
        )
        provenance.append(_provenance_entry(exported, output_root))
    wheel = _build_wheel(output_root)
    _copy_common_files(output_root, wheel)
    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "release_id": release["release_id"],
        "source_repository": release["source_repository"],
        "source_revision": source_revision,
        "model_type": "cspine-segmentation",
        "labels": ["background", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "models": models,
        "preprocessing": {
            "input": "one physically ordered axial CT series",
            "window_center": 400.0,
            "window_width": 1800.0,
            "window_hu": [-500.0, 1300.0],
            "quantization": "round_to_uint8",
            "input_shape": [128, 128, 128],
            "resize": "trilinear",
            "output_native_resize": "nearest",
            "component_filter": "largest_3d_component_per_level",
        },
        "ensemble": {"method": "mean_softmax_probability", "folds": 5},
        "python_wheel": wheel,
    }
    _write_json(output_root / "cspine_model.json", manifest)
    _write_json(
        output_root / "provenance.json",
        {
            "release_id": release["release_id"],
            "source_repository": release["source_repository"],
            "source_revision": source_revision,
            "checkpoints": provenance,
        },
    )
    (output_root / "inference.py").write_text(
        "from skp.cspine.pipeline import CervicalSpineSegmenter\n\n"
        "model = CervicalSpineSegmenter.from_pretrained('.', device='cuda')\n"
    )
    (output_root / "README.md").write_text(_segmenter_card())
    return output_root


def export_classifier(
    output_root: Path,
    *,
    overwrite: bool,
    experiment_root: Path = EXPERIMENT_ROOT,
    series_root: Path = SERIES_ROOT,
    release_manifest: Path = DEFAULT_RELEASE_MANIFEST,
    source_revision: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    release = load_release_manifest(release_manifest)
    source_revision = source_revision or release["source_revision"]
    family_specs = release["selected_pipeline"]["families"]
    families = {}
    provenance = []
    for family_name, spec in family_specs.items():
        folds = []
        for fold in range(5):
            components = {}
            sources = {
                "mil": (experiment_root, spec["mil"]),
                "7x1": (series_root, spec["series_7x1"]),
                "7x32": (series_root, spec["series_7x32"]),
                "localizer": (experiment_root, spec["localizer"]),
            }
            for component, (artifact_root, component_specification) in sources.items():
                checkpoint = best_checkpoint(
                    artifact_root,
                    component_specification,
                    fold=fold,
                )
                output_path = (
                    output_root
                    / "weights"
                    / family_name
                    / component
                    / f"fold{fold}.safetensors"
                )
                exported = _export_checkpoint(
                    checkpoint,
                    output_path,
                    use_ema=component_specification["checkpoint_state"] == "ema",
                    overwrite=overwrite,
                    source_roots=(
                        (experiment_root, "project-experiments"),
                        (series_root, "series-experiments"),
                    ),
                )
                components[component] = _relative_entry(
                    exported,
                    output_root,
                    config=component_specification["config"],
                    fold=fold,
                )
                provenance.append(_provenance_entry(exported, output_root))
            folds.append(components)
        families[family_name] = {
            "crop_tensor": spec["crop_tensor"],
            "folds": folds,
        }
    wheel = _build_wheel(output_root)
    _copy_common_files(output_root, wheel)
    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "release_id": release["release_id"],
        "source_repository": release["source_repository"],
        "source_revision": source_revision,
        "model_type": "cspine-fracture-classification",
        "labels": ["overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        "requires_model_type": "cspine-segmentation",
        "families": families,
        "preprocessing": {
            "initial_level_padding_fraction": 0.25,
            "close_mask_threshold": 0.4,
            "close_padding_fraction": 0.05,
            "axial_shape": [384, 384],
            "d32_depth": 32,
            "stack3_source_depth": 96,
            "window_center": 400.0,
            "window_width": 1800.0,
        },
        "ensemble": {
            "folds": 5,
            "architectures": ["7x1_d32", "7x1_3x32", "7x32_d32", "7x32_3x32"],
            "method": "equal_mean_probability",
            "oof_competition_log_loss": release["reported_oof_metrics"][
                "competition_weighted_log_loss"
            ],
            "oof_auc_overall": release["reported_oof_metrics"]["overall_auc"],
            "oof_auc_levels_mean": release["reported_oof_metrics"][
                "macro_c1_c7_auc"
            ],
        },
        "localization": {
            "families": ["d32", "3x32"],
            "method": "equal_mean_probability_on_96x96x96_crop_grid",
            "default_level_threshold": 0.5,
        },
        "inference": {
            "encoder_chunk_size": 32,
            "crop_batch_size": 1,
            "localization_batch_size": 1,
        },
        "python_wheel": wheel,
    }
    _write_json(output_root / "cspine_model.json", manifest)
    _write_json(
        output_root / "provenance.json",
        {
            "release_id": release["release_id"],
            "source_repository": release["source_repository"],
            "source_revision": source_revision,
            "checkpoints": provenance,
        },
    )
    (output_root / "inference.py").write_text(
        "from skp.cspine.pipeline import CervicalSpineFracturePredictor\n\n"
        "model = CervicalSpineFracturePredictor.from_pretrained('.', device='cuda')\n"
    )
    (output_root / "README.md").write_text(_classifier_card())
    return output_root


def _segmenter_card() -> str:
    return """---
library_name: skp
pipeline_tag: image-segmentation
license: apache-2.0
tags:
  - medical-imaging
  - computed-tomography
  - cervical-spine
  - safetensors
---

# Cervical spine C1-C7 segmentation

Five-fold X3D-S/DeepLabV3+ ensemble trained with TotalSegmentator pseudo-label
pretraining followed by manual-label fine-tuning. The model consumes one axial
CT series windowed at center 400 / width 1800 and predicts C1-C7 masks.

This is a research model, not a medical device, and must not be used as the
sole basis for diagnosis or treatment.

## Local installation and use

```bash
bash install_local.sh
```

```python
from skp.cspine.pipeline import DICOMSeries, CervicalSpineSegmenter

series = DICOMSeries.from_dicom_directory("/path/to/one/series")
segmenter = CervicalSpineSegmenter.from_pretrained(".", device="cuda")
segmentation = segmenter(series)
print(segmentation.anomaly_levels)
```

See `cspine_model.json` for the machine-readable preprocessing and ensemble
contract. The companion fracture-classification repository consumes this
model's `SegmentationResult`.

## Evaluation

Mean fold-best C1-C7 Dice: 0.8078 on the small manually segmented subset.
Full-resolution TotalSegmentator remains the teacher-quality ceiling.

## Limitations

Training is based on the RSNA 2022 cervical spine fracture competition data.
Performance may degrade under domain shift, incomplete cervical coverage,
severe artifacts, altered reconstructions, or thick slices. Topology anomaly
flags must be inspected downstream.
"""


def _classifier_card() -> str:
    return """---
library_name: skp
pipeline_tag: image-classification
license: apache-2.0
tags:
  - medical-imaging
  - computed-tomography
  - cervical-spine-fracture
  - multiple-instance-learning
  - safetensors
---

# Cervical spine fracture classification and localization

Five-fold d32 and 3x32 MaxViT-Tiny MIL families with four lightweight series
heads (7x1 and 7x32 for each family). The package predicts overall fracture and
C1-C7 probabilities and includes binary d32/3x32 localization decoders.

This is a research model, not a medical device. Heatmaps are weak/localization
supervision outputs and are not pixel-accurate fracture segmentations.

## End-to-end use with the segmentation model

Install this repository's local wheel and requirements, then obtain the
companion segmentation model directory:

```bash
bash install_local.sh
```

```python
from skp.cspine.pipeline import (
    CervicalSpineFracturePredictor,
    CervicalSpinePipeline,
    CervicalSpineSegmenter,
)

segmenter = CervicalSpineSegmenter.from_pretrained(
    "/path/to/cspine-segmentation", device="cuda"
)
predictor = CervicalSpineFracturePredictor.from_pretrained(".", device="cuda")
pipeline = CervicalSpinePipeline(segmenter, predictor)

series, segmentation, result = pipeline("/path/to/one/dicom/series")
result.save("./result")
print(result.probabilities)
```

`prediction.json` contains eight probabilities, architecture-level predictions,
quality flags, and exact native slice geometry. Each localized level also gets
a compressed 96x96x96 heatmap file. Restore a localization to the source DICOM
coordinate system without retaining a dense 3D native-volume array:

```python
for slice_index, instance_number, heatmap_2d in (
    result.localizations["C2"].iter_native_heatmaps(series.volume.shape)
):
    # Overlay heatmap_2d on series.volume[slice_index].
    pass
```

## Five-fold OOF evaluation

- Competition weighted log loss: 0.25981
- Overall fracture AUC: 0.93930
- Macro C1-C7 AUC: 0.94087

## Limitations

Training is based on RSNA 2022 competition data. Thick-slice robustness has not
yet been specifically trained or validated. Missing vertebrae and topology
anomalies are surfaced as quality flags. Classification and heatmaps must be
reviewed by qualified clinicians and must not replace diagnostic interpretation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export two Hugging Face-ready cervical spine model directories."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/huggingface/final-v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Root containing segmentation, MIL, and localization experiment runs.",
    )
    parser.add_argument(
        "--series-root",
        type=Path,
        default=SERIES_ROOT,
        help="Root containing the four selected series-head experiment runs.",
    )
    parser.add_argument(
        "--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST
    )
    parser.add_argument(
        "--source-revision",
        help="Public Git commit or immutable tag recorded in exported provenance.",
    )
    args = parser.parse_args()
    segmentation = export_segmenter(
        args.output_root / "cspine-segmentation",
        overwrite=args.overwrite,
        experiment_root=args.experiment_root,
        release_manifest=args.release_manifest,
        source_revision=args.source_revision,
    )
    classifier = export_classifier(
        args.output_root / "cspine-fracture-classifier",
        overwrite=args.overwrite,
        experiment_root=args.experiment_root,
        series_root=args.series_root,
        release_manifest=args.release_manifest,
        source_revision=args.source_revision,
    )
    print(f"segmentation={segmentation}")
    print(f"classifier={classifier}")


if __name__ == "__main__":
    main()
