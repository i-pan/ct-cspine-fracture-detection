# CT Cervical Spine Fracture Detection

Reproducible training and inference for vertebral segmentation, cervical-spine
fracture classification, and conditional fracture localization on the RSNA
2022 CT dataset.

The final system predicts eight probabilities—overall fracture and C1 through
C7—and produces a heatmap for a predicted fractured vertebra. It is split into
two independently loadable model packages:

- **C1-C7 segmentation:** a five-fold X3D-S ensemble used to identify and crop
  individual vertebrae;
- **fracture prediction and localization:** d32 and 3x32 MaxViT-Tiny MIL
  families, four series-level sequence heads, and two binary localization
  decoder ensembles.

This is research software, not a medical device. It has not been externally or
clinically validated and must not be used as the sole basis for diagnosis,
triage, or treatment.

## Results

Five-fold out-of-fold evaluation on all 2,019 competition training studies:

| Metric | Final ensemble |
|---|---:|
| Competition weighted log loss | **0.25981** |
| Overall fracture AUC | **0.93930** |
| Macro C1-C7 AUC | **0.94087** |

The final prediction is the equal probability mean of four raw/no-projection
series heads: d32 7x1, d32 7x32, 3x32 7x1, and 3x32 7x32. The public OOF CLI
reproduces the historical AUCs exactly and the weighted loss within `3e-10`.

See [results](docs/RESULTS.md) and the complete
[experiment history](docs/EXPERIMENT_HISTORY.md).

## Architecture

```text
DICOM CT
  -> WL400/WW1800 windowing
  -> C1-C7 segmentation ensemble
  -> native-depth C1-C7 crops
  -> d32 + 3x32 vertebra MIL models
  -> 7x1 + 7x32 series heads
  -> overall/C1-C7 probabilities
  -> conditional d32 + 3x32 fracture heatmaps
```

The classifier is image-only. It does not use acquisition metadata. Individual
MIL inputs contain three CT channels and the target vertebra's soft mask.

See the detailed [pipeline contract](docs/PIPELINE.md) and the authoritative
machine-readable [`final-v1` manifest](release/final-v1.json).

## Install

Python 3.11 and CUDA 12.4 were used for the reference runs.

```bash
git clone https://github.com/i-pan/ct-cspine-fracture-detection.git
cd ct-cspine-fracture-detection
uv sync --locked --group train --group dev --group medical --group video
```

## End-to-end inference

Once the two Hugging Face model directories are available:

```bash
uv run cspine-predict /path/to/one/dicom-series \
  --segmenter /path/or/hf-id/cspine-segmentation \
  --predictor /path/or/hf-id/cspine-fracture-classifier \
  --output ./prediction
```

Python usage:

```python
from skp.cspine.pipeline import (
    CervicalSpineFracturePredictor,
    CervicalSpinePipeline,
    CervicalSpineSegmenter,
)

segmenter = CervicalSpineSegmenter.from_pretrained(
    "/path/or/hf-id/cspine-segmentation", device="cuda"
)
predictor = CervicalSpineFracturePredictor.from_pretrained(
    "/path/or/hf-id/cspine-fracture-classifier", device="cuda"
)
pipeline = CervicalSpinePipeline(segmenter, predictor)
series, segmentation, result = pipeline("/path/to/one/dicom-series")
result.save("./prediction")
print(result.probabilities)
```

See [inference documentation](docs/INFERENCE.md) for heatmap restoration and
the supported input contract.

## Full reproduction

Users must accept the Kaggle competition rules and download the public dataset
themselves. No DICOMs, PNGs, NIfTI files, masks, or generated annotations are
distributed in this repository.

The complete path from the Kaggle data to the final two Hugging Face packages
is documented in [REPRODUCE.md](docs/REPRODUCE.md). It covers:

1. deterministic DICOM windowing and study-level folds;
2. a pinned TotalSegmentator 2.16.0 teacher environment;
3. segmentation pseudo-label pretraining and manual fine-tuning;
4. fold-safe OOF masks and native-depth vertebral crops;
5. d32 and 3x32 MIL training;
6. localization training;
7. frozen feature extraction and four series heads;
8. OOF metric reproduction and SafeTensor export.

Dataset layout, expected counts, and leakage constraints are in
[DATA.md](docs/DATA.md).

## Reproducibility notes

- `release/final-v1.json` selects every config, run, checkpoint state, fold
  exception, and ensemble member.
- Segmentation, MIL, and localization release weights use EMA; series heads use
  their raw model state.
- Pretrained X3D code comes from the lockfile-pinned `pytorchvideo` package,
  rather than mutable Torch Hub source. Initialization hashes are recorded.
- Published SafeTensors are hashed and verified before loading.
- The original research runs did not consistently set random seeds. Published
  weights are exact artifacts; fresh retraining is expected to reproduce
  performance within normal stochastic variation, not identical file hashes.
- Normal training requires a reachable MLflow server and fails closed if it is
  unavailable.

## Development

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

Project-specific reasoning is recorded in
[DECISION_LOG.md](docs/DECISION_LOG.md). Historical and ablation configs remain
available under `src/skp/configs/`; only components named by the release
manifest are part of `final-v1`.

## License and data terms

Source code is licensed under [Apache-2.0](LICENSE). The RSNA/Kaggle dataset is
governed separately by the competition rules and is not redistributed here.
Model weights and outputs are for research use and inherit the limitations
described in their model cards.
