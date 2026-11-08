# Reproduce final-v1

This guide rebuilds the selected pipeline from the user-supplied RSNA/Kaggle
dataset. It does not reproduce data or checkpoints byte-for-byte: the original
training runs did not consistently set seeds and CUDA kernels are not fully
deterministic. Published artifacts are verified by SHA-256; retraining should
be compared with the documented cross-validation metrics and artifact counts.

All commands run from the repository root.

## 1. Environment and storage

```bash
git clone https://github.com/i-pan/ct-cspine-fracture-detection.git
cd ct-cspine-fracture-detection
uv sync --locked --group train --group dev --group medical --group video
cp .env.example .env
```

Set these values in `.env`:

```dotenv
CSPINE_DATA_ROOT=/path/to/rsna-2022-cervical-spine-fracture-detection
CSPINE_WORK_ROOT=/large-filesystem/rsna-2022-cspine-processed
CSPINE_EXPERIMENT_ROOT=/large-filesystem/cspine-experiments
MLFLOW_TRACKING_URI=http://your-reachable-mlflow-server:5001
```

Load the path variables into the current shell as well as the Python process:

```bash
set -a
source .env
set +a
```

Training is fail-closed when MLflow is missing or unreachable. The reference
workstation used two 48 GB RTX A6000 GPUs, BF16, and DDP. Commands below add a
seed for future reproduction even though the original runs were unseeded.

Create the isolated teacher environment:

```bash
scripts/setup_totalsegmentator.sh \
  "$CSPINE_WORK_ROOT/tools/totalsegmentator-2.16.0"
```

The pinned teacher environment targets CUDA 12.4. On first use,
TotalSegmentator downloads its `total` task weights. Their expected Dataset292
hashes are recorded in `release/final-v1.json`.

## 2. PNGs, folds, and manual segmentation cache

```bash
uv run cspine-convert-dicom \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT" \
  --workers 12

uv run cspine-build-manifest \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT" \
  --folds 5 --seed 2022

uv run cspine-build-segmentation-cache \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT"
```

The converter maps HU `[-500, 1300]` to rounded uint8 PNGs without resizing.
Verify the counts in `docs/DATA.md` before training.

## 3. TotalSegmentator teacher and segmentation student

Benchmark the teacher on the 87 manual cases, then pseudo-label the remaining
studies. Temporary NIfTI files are created per study and deleted immediately.

```bash
TS_ENV="$CSPINE_WORK_ROOT/tools/totalsegmentator-2.16.0"

uv run cspine-totalsegmentator-teacher \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT" \
  --environment "$TS_ENV" --devices 0,1 --manual-only

uv run cspine-totalsegmentator-teacher \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT" \
  --environment "$TS_ENV" --devices 0,1
```

The second command writes both the 1,932-row raw manifest and deterministic
1,931-row QC manifest.

```bash
uv run skp-train cspine_segmentation_x3d_s_totalseg_pretrain \
  --kfold 0,1,2,3,4 --run_id cspine-seg-x3ds-ts-pretrain-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed

uv run skp-train cspine_segmentation_x3d_s_totalseg_finetune \
  --kfold 0,1,2,3,4 --run_id cspine-seg-x3ds-ts-finetune-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed
```

The trainer rewrites the pretraining checkpoint path from `fold0` to the active
fold, so every fine-tuning fold uses its matching pseudo-label-pretrained model.

Confirm the release checkpoint-state policy when GPUs are available:

```bash
uv run cspine-evaluate-segmentation-states \
  --release-manifest release/final-v1.json \
  --experiment-root experiments \
  --output "$CSPINE_EXPERIMENT_ROOT/segmentation-state-evaluation.json"
```

Validation and checkpoint selection used EMA. This comparison records the
matched raw result rather than attributing the EMA Dice to raw weights.

## 4. OOF masks and native-depth vertebral crops

```bash
SEG_RUN="experiments/cspine_segmentation_x3d_s_totalseg_finetune/cspine-seg-x3ds-ts-finetune-v1"

uv run cspine-infer-segmenter \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT" \
  --config cspine_segmentation_x3d_s_totalseg_finetune \
  --checkpoint-fold 0="$SEG_RUN/fold0/checkpoints/best.ckpt" \
  --checkpoint-fold 1="$SEG_RUN/fold1/checkpoints/best.ckpt" \
  --checkpoint-fold 2="$SEG_RUN/fold2/checkpoints/best.ckpt" \
  --checkpoint-fold 3="$SEG_RUN/fold3/checkpoints/best.ckpt" \
  --checkpoint-fold 4="$SEG_RUN/fold4/checkpoints/best.ckpt" \
  --ensemble-on-qc-failure --compact-grid

uv run cspine-build-crops \
  --data-root "$CSPINE_DATA_ROOT" \
  --output-root "$CSPINE_WORK_ROOT" \
  --mask-root "$CSPINE_WORK_ROOT/vertebra_masks/oof" \
  --shape 512 384 384 --variable-depth \
  --crop-subdir vertebra_crops_native_depth \
  --manifest-name vertebra_samples_native_depth.parquet \
  --exclude-unresolved-qc --allow-missing \
  --qc-teacher-annotations \
    "$CSPINE_WORK_ROOT/manifests/totalsegmentator_samples_qc.parquet"
```

Depth 512 is only a cap; `--variable-depth` preserves every selected crop at
its actual depth. The observed maximum is 348 slices.

## 5. Selected d32 and 3x32 MIL models

```bash
uv run skp-train cspine_vertebra_mil_maxvit_tiny_d32_close_square \
  --kfold 0,1,2,3,4 \
  --run_id cspine-vertebra-maxvitt-mil-d32-close-square-imageonly-bs2-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed

uv run skp-train cspine_vertebra_mil_maxvit_tiny_d32_stack3 \
  --kfold 0,1,2,3 \
  --run_id cspine-vertebra-maxvitt-mil-d32-stack3x96-imageonly-bs2-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed

uv run skp-train cspine_vertebra_mil_maxvit_tiny_d32_stack3_loaderfix \
  --kfold 4 \
  --run_id cspine-vertebra-maxvitt-mil-d32-stack3x96-imageonly-fold4-loaderfix-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed
```

Fold 4 uses the same model and optimization with a bounded, non-persistent
loader to avoid the input-pipeline slowdown observed in the original queue.

## 6. Localization decoders

```bash
uv run cspine-build-localization-manifest \
  --data-root "$CSPINE_DATA_ROOT" --output-root "$CSPINE_WORK_ROOT" \
  --mask-root "$CSPINE_WORK_ROOT/vertebra_masks/oof" \
  --target-depth 32 --manifest-name fracture_localization_d32.parquet

uv run cspine-build-localization-manifest \
  --data-root "$CSPINE_DATA_ROOT" --output-root "$CSPINE_WORK_ROOT" \
  --mask-root "$CSPINE_WORK_ROOT/vertebra_masks/oof" \
  --target-depth 96 --manifest-name fracture_localization_3x32.parquet

uv run skp-train cspine_fracture_localization_d32 \
  --kfold 0,1,2,3,4 --run_id cspine-localize-d32-frozen-mil-binary-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed

uv run skp-train cspine_fracture_localization_3x32 \
  --kfold 0,1,2,3,4 --run_id cspine-localize-3x32-frozen-mil-binary-v1 \
  --seed 2022 --accelerator cuda --devices 2 --strategy ddp \
  --precision bf16-mixed
```

## 7. Frozen features and four series heads

Generate pooled and depth-resolved features from every fold-specific EMA MIL
model:

```bash
for FAMILY in d32 3x32; do
  uv run cspine-extract-series-features --family "$FAMILY" \
    --representation pooled \
    --experiment-root experiments \
    --output-root "$CSPINE_WORK_ROOT/series_features"
  uv run cspine-extract-series-features --family "$FAMILY" \
    --representation depth \
    --experiment-root experiments \
    --output-root "$CSPINE_WORK_ROOT/series_features_7x32"
done
```

Train the four no-projection heads. These are lightweight and can be trained on
CPU, although CUDA is faster.

```bash
uv run skp-train cspine_series_d32_no_projection --kfold 0,1,2,3,4 \
  --run_id series-d32-raw-v1 --seed 2022 --accelerator cpu \
  --devices 1 --strategy auto --no_sync_batchnorm

uv run skp-train cspine_series_3x32_no_projection --kfold 0,1,2,3,4 \
  --run_id series-3x32-raw-v1 --seed 2022 --accelerator cpu \
  --devices 1 --strategy auto --no_sync_batchnorm

uv run skp-train cspine_series_7x32_d32_no_projection --kfold 0,1,2,3,4 \
  --run_id series-7x32-d32-raw-v1 --seed 2022 --accelerator cpu \
  --devices 1 --strategy auto --no_sync_batchnorm

uv run skp-train cspine_series_7x32_3x32_no_projection --kfold 0,1,2,3,4 \
  --run_id series-7x32-3x32-raw-v1 --seed 2022 --accelerator cpu \
  --devices 1 --strategy auto --no_sync_batchnorm
```

## 8. OOF verification and HF export

```bash
uv run cspine-evaluate-series-oof \
  --release-manifest release/final-v1.json \
  --series-experiment-root "$CSPINE_EXPERIMENT_ROOT" \
  --pooled-feature-root "$CSPINE_WORK_ROOT/series_features" \
  --depth-feature-root "$CSPINE_WORK_ROOT/series_features_7x32" \
  --output-root "$CSPINE_EXPERIMENT_ROOT/series_ensemble/final-v1"
```

Expected final metrics are 0.25981 weighted log loss, 0.93930 overall AUC,
and 0.94087 macro C1-C7 AUC.

```bash
uv run cspine-export-hf \
  --release-manifest release/final-v1.json \
  --experiment-root experiments \
  --series-root "$CSPINE_EXPERIMENT_ROOT" \
  --source-revision "$(git rev-parse HEAD)" \
  --output-root "$CSPINE_WORK_ROOT/huggingface/final-v1"
```

Export fails if outputs already exist unless `--overwrite` is supplied. Every
SafeTensor is hashed and inference verifies the manifest checksum before model
loading.
