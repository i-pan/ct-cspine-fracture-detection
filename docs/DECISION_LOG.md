<!-- skpv2-project-documentation: decision-log -->

# Project Decision Log

This file records detailed, project-specific decisions and analysis. Each
material decision should capture its context, evidence, alternatives,
tradeoffs, conclusion, and consequences.

> **This file is not `DECISIONS.md`.**
>
> The repository-root [`DECISIONS.md`](../DECISIONS.md) records decisions about
> developing and maintaining the reusable **SKPv2 template/framework**.
> `docs/DECISION_LOG.md` records decisions about the **experiment or product
> being built from that template**. Do not put modeling, dataset, evaluation, or
> release decisions for the current project into the template decision history.

This document is complementary to
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md), and intentional overlap is
expected:

- **Experiment History** is primarily an experiment tracker: what ran, how it
  was configured, what happened, and whether it was retained.
- **Decision Log** provides the more detailed reasoning and analysis: why the
  experiment was run, how evidence was interpreted, which alternatives were
  considered, and what the result changes downstream.

An important experiment will often have a concise row or section in Experiment
History and a fuller entry here.

## 2026-07-28: Initial segmentation and vertebra-classification milestone

### Context

The project needs accurate C1-C7 localization and fracture classification while
also retaining a path to slice-level localization and thicker-slice robustness.
The supplied data contains 2,019 studies, 87 C1-C7 NIfTI masks, and sparse
fracture boxes.

### Evidence

The top RSNA 2022 solutions consistently localized vertebral anatomy before
classification. The public third-place implementation used a WL400/WW1800 bone
window and adjacent slices as 2.5D channels. Local inspection found 711,601
training DICOMs, 961 positive studies, 1,444 positive vertebra labels, variable
study depth, and slope/intercept encodings that must be applied before
windowing.

### Alternatives considered

- Isotropic-spacing resampling was rejected for the first milestone in favor of
  fixed dimensions.
- End-to-end study classification was deferred because separate vertebra crops
  provide roughly 14,000 samples and less ambiguous weak labels.
- Fracture heatmaps, pseudo-slice labels, patient aggregation, and thick-slice
  augmentation were designed but deferred from active training.

### Decision

- Convert DICOMs to native-resolution uint8 PNG using fixed WL400/WW1800.
- Train five-fold C1-C7 segmentation at a fixed 128-cube input.
- Export fold-safe masks and crop each vertebra to `32 x 256 x 256`.
- Train shared, level-aware ConvNeXt-Tiny and EfficientNetV2-S CNN-BiGRU MIL
  binary classifiers.

### Consequences and follow-up

All derived samples inherit the study fold. Classification training must use
out-of-fold segmentation masks. The next milestone begins only after these two
tasks have reliable cross-validated results.

## 2026-07-28: Full-resolution TotalSegmentator pseudo-label pretraining

### Context

Only 87 of 2,019 training studies have manual C1-C7 masks. The manual-only
student has adequate localization performance but leaves most available CT
anatomy unused.

### Evidence

The five-fold X3D-S manual-only baseline retained best mean Dice values of
0.6660, 0.6742, 0.6474, 0.6983, and 0.6867. Full-resolution
TotalSegmentator 2.16.0 was then evaluated against every manual mask and
achieved 0.8940 +/- 0.0170 mean study Dice with no missing C1-C7 labels.
Performance did not degrade in fracture-positive studies (0.8961 versus
0.8922 in negatives).

The newer `total_v3` task exposed by the 2.16.0 CLI was investigated, but its
referenced upstream `v3.0.0-weights` release and weight assets returned 404.
It is therefore not a reproducible model option at this time.

### Alternatives considered

- The 3 mm `--fast` model was rejected because the objective is the strongest
  available teacher, especially at C1/C7 boundaries.
- ROI-only inference was rejected because its crop localizer can reduce
  accuracy and the full task is tractable on two A6000 GPUs.
- Persisting a second native CT/NIfTI cache was rejected because it would
  consume substantial storage without helping student training.
- Training a common pseudo-label model on images from a held-out fold was
  rejected as transductive cross-validation leakage.

### Decision

Use the strongest published, runnable open teacher: TotalSegmentator 2.16.0
`total` at its full 1.5 mm resolution with cubic input resampling. Do not use
fast mode, ROI cropping, or small-component removal. For each study:

1. create an HU NIfTI in a scoped temporary directory using the DICOM
   orientation, position, spacing, and LPS-to-RAS affine;
2. run a single multilabel TotalSegmentator output;
3. dynamically remap the installed class IDs to C1-C7;
4. retain only a compressed fixed `128^3` student image/mask and a QC record;
5. delete the temporary input/output NIfTIs immediately.

### Consequences and follow-up

Pseudo-label generation is resumable per study and bounded to two in-flight
temporary workspaces. For each validation fold, pseudo-label pretraining must
exclude every study assigned to that fold, even if it lacks a manual mask.
The pretrained model will then be fine-tuned on manual masks from the four
training folds and evaluated only on the held-out manual fold. The
manual-only result remains the required comparator.

## 2026-08-01: Compact fixed-volume 3D Halton search

### Context

Depth-preserving X3D with global D/H/W pooling improved much faster than the
slice-LSE 3D variants, while X3D-L and CSN-26 had not received a fair search
over input geometry, batch size, learning rate, and longer training horizons.
The search needs to remain small enough to finish in roughly one day.

### Evidence

The X3D-S `64x288x288` global-pooling anchor reached fold-0 macro AUC 0.8409
after epoch 6 and had not clearly plateaued. Earlier 4-8 epoch proposals were
therefore too short. The three selected shapes use approximately equal input
voxel counts. Creating dedicated copies would require tens of GiB per shape,
while only 19 GiB remained on the processed-data volume. Exact runtime
resampling from native-depth, 384-pixel fitted crops measured about 0.12
seconds per sample before dataloader parallelism.

### Decision

Use the existing run as an anchor and train five deterministic, coverage-aware
points from a shifted Halton sequence. Search backbone, coupled input shape,
batch size/GPU, maximum epochs, and log-uniform learning rate. Fix global 3D
average pooling, all-one temporal strides, CT plus level-specific soft mask,
weighted sampling, unweighted BCE, SyncBatchNorm, EMA, AdamW weight decay
0.01, BF16 DDP, and no gradient accumulation. Match cosine schedule duration
to each 8/12/16-epoch trial and use one warm-up epoch.

Run two-batch memory smoke tests before training. If batch 4 OOMs, retry batch
2 rather than accumulating gradients. Prune only at matched epoch 4 when more
than 0.05 behind the incumbent or epoch 8 when more than 0.03 behind. Retain
per-epoch checkpoints and metrics so optimal epoch can be assessed separately
from the sampled maximum.

### Consequences and follow-up

Runtime resampling avoids new persistent crop caches and reproduces the crop
builder's centered fit-without-magnification rule. Queue and pruning state are
stored under `experiments/cspine_3d_halton_search`, while every real training
run remains in the shared classification MLflow experiment. Rank by fold-0
macro AUC, then overall AUC, C6/C7 performance, and runtime. Confirm the
winner on fold 1; consider `64x384x384` only if the 320-pixel condition wins.

## 2026-08-01: Reproduce the original X3D-L chunk classifier before searching

### Context

The compact Halton search was tuning a materially different pipeline from the
original 6th-place 3D classifier. Differences in localization crop geometry,
input channels, resampling, sampling weights, optimization, and validation
splits made it difficult to determine whether backbone hyperparameters or the
pipeline contract explained the performance gap.

### Evidence

The original `chunk000` code uses tight threshold-0.4 vertebra coordinates,
nearest-neighbor index zoom to 64 slices, aspect-ratio-agnostic linear resize
to 288 square, grayscale input normalized to `[-1,1]`, X3D-L with every depth
stride equal to one, global average pooling, five-sample dropout, natural
sampling, unweighted BCE, batch 4/GPU, 10 epochs, AdamW 3e-4/5e-4, cosine
decay, SyncBatchNorm, fp16, seed 88, and non-stratified GroupKFold. Its metric
named `auc_mean` is one overall binary AUC across vertebra samples rather than
a macro average of per-level AUCs. The current Halton contract instead used
padded scale-preserving crops, CT plus mask input, weighted sampling, EMA,
BF16, stronger weight decay, warmup, and multilabel-stratified folds.

### Decision

Stop the remaining Halton trials and run a fold-0 X3D-L reproduction with the
original contract wherever the current artifacts allow. Implement the crop
and depth sampler at dataset load time so no additional full cache is needed,
and recreate the original study folds in a dedicated manifest. Rank the run
by overall AUC while retaining macro and per-level AUCs as diagnostics.

### Consequences and follow-up

The first comparison isolates most pipeline differences without deleting any
existing cache. Two source differences remain explicit: current PNGs use
WL400/WW1800 rather than the original WW2500, and localization uses current
OOF student masks rather than the original five-model segmentation ensemble.
With only 19 GiB free, a second PNG/cache copy is deferred. If the reproduction
recovers the expected performance, future ablations should start from this
contract and change one component at a time; otherwise, windowing and exact
localizer outputs are the next reproduction gaps to close.

## 2026-08-02: Select image-only close-square MaxViT candidates for five-fold validation

### Context

The initial 384-pixel MaxViT MIL model was promising, but its padded crop
geometry wasted resolution and its sequence model received acquisition
metadata and a learned C1-C7 identity. A high fold-0 score after only two
epochs raised a legitimate concern about split leakage or metadata shortcuts.
The project also needed to compare ordinary overlapping adjacent-slice inputs
against a 96-slice volume reshaped into 32 non-overlapping triplets.

### Evidence

The manifest contains 403 validation studies and 2,821 vertebra samples in
fold 0. No study, crop path, or study-level pair overlaps the training split,
and each study belongs to exactly one fold. Validation is deterministic and
per-level AUC correctly filters samples by C1-C7 before macro averaging.
Simple acquisition variables were weak individually; the strongest checked
univariate AUC was 0.5884 for mean slice spacing.

The close-square crop thresholds the target-level mask at 0.4 with fallback to
0.1, crops to the mask's depth bounds, adds 5% padding around the larger axial
extent, shifts the resulting square at image boundaries, and resizes
square-to-square at 384 pixels. Depth downsampling is trilinear and upsampling
uses nearest-neighbor repetition.

The metadata-enabled depth-32 run reached 0.9299 macro and 0.9392 overall AUC.
After removing acquisition metadata and the C1-C7 embedding entirely, the
matched image-only run improved to 0.9328 macro and 0.9439 overall AUC at epoch
4. The model input is now limited to previous/current/next windowed CT and the
target vertebra's soft mask. Changing the metadata tensor and level index is
covered by a regression test and cannot change model output.

The matched 96-to-3x32 image-only run reached 0.9312 macro and 0.9475 overall
AUC at epoch 5. Thus depth 32 leads macro AUC by 0.0016, while 3x32 leads
overall AUC by 0.0036. Both gaps are too small to settle from one fold.

### Alternatives considered

- Retaining acquisition metadata was rejected because the intended classifier
  should learn from the localized CT and mask rather than scanner geometry or
  acquisition protocol.
- A learned C1-C7 embedding is a reasonable later ablation, but it was
  cancelled to prioritize cross-validating the two strongest image-only
  formulations.
- Selecting only depth 32 would optimize the configured macro checkpoint
  metric, while selecting only 3x32 would optimize overall binary AUC. Neither
  single-fold distinction is strong enough to discard the other model.
- The interrupted 512-pixel run is not used to decide resolution because its
  backbone was the 384-pretrained MaxViT variant.

### Decision

Train both image-only candidates on folds 1-4 with the same study-level folds,
weighted sampler, unweighted BCE, batch 2/GPU, no accumulation, eight-epoch
training horizon, and BF16 two-GPU DDP. Keep acquisition metadata and level
identity out of the forward pass. Rank final candidates using cross-validated
macro and overall AUC together, inspect per-level dispersion, and retain OOF
predictions for correlation and ensemble analysis.

### Consequences and follow-up

Five-fold classification selected depth 32: mean overall/macro AUC was
0.9434/0.9350 versus 0.9393/0.9299 for 3x32. The close-square result therefore
makes depth 32 the primary classification and preprocessing baseline.

The two models are nevertheless complementary for localization. Across all 235
out-of-fold boxed studies, mean per-fold slice AUC/AP was 0.8688/0.6902 for
depth 32, 0.8644/0.6833 for 3x32, and 0.8905/0.7366 for their native-grid logit
ensemble. The ensemble improved both metrics on every fold and is selected as
the slice-pseudolabel teacher. It covered 99.74% of annotated slices. Report
mean fold metrics as the primary CV estimate because raw logits from separate
fold models are not identically calibrated; retain raw pooled metrics as a
secondary diagnostic.

The native-grid ensemble generated 448,485 soft OOF slice targets. A single
mask-free MaxViT-Tiny 384 student using previous/current/next slices as its
three channels peaked at epoch 0 with mean-fold AUC/AP 0.9118/0.7802 and pooled
AUC/AP 0.9104/0.7770, improving both metrics over the teacher in every fold.
Later epochs degraded, so retain the epoch-0 checkpoint and use a one-epoch
effective training horizon for follow-up slice distillation. Manual labels
were evaluation-only, but this single student trained on OOF targets for all
images, including the boxed subset; its score is therefore not a fold-excluded
student generalization estimate.

The next localization milestone is to initialize the MIL encoder from the
selected slice checkpoint and test whether warmup preserves or improves the
vertebra-level result. Thick-slice robustness remains subsequent work.

## 2026-08-07: Initialize d32 MIL from the mask-free slice student

### Evidence and decision

The fold-0 transfer run copied all three adjacent-slice encoder channels,
zero-initialized the new mask channel, and fine-tuned for two epochs using a
10% warmup and cosine LR schedule from 2e-4 to 2e-6. It reached overall/macro
AUC 0.9473/0.9359, improving on the original eight-epoch fold-0 best of
0.9439/0.9328. At matched epoch 1 the improvement was 0.0170/0.0177. Retain
the slice initialization and two-epoch schedule for the remaining folds.

Across all five folds, d32 transfer reached overall/macro AUC
0.9450/0.9382 versus 0.9434/0.9350 without transfer. Macro AUC improved on
three folds and regressed on two, making this a modest mean improvement with a
large training-time reduction rather than a universal per-fold gain. The same
experiment on 3x32 reached 0.9372/0.9277 versus its original 0.9393/0.9299.
Use slice initialization for d32, but reject it for 3x32.

This is a transductive encoder initialization experiment: the slice student
never used manual fracture targets, but it was optimized on OOF teacher
predictions for all study images. Do not describe the resulting fold metrics
as a fully fold-excluded estimate.

## 2026-08-08: Train a binary heatmap decoder on the frozen original MIL encoder

### Context

The vertebra MIL classifiers identify whether an individual crop is fractured,
but weak slice logits do not provide a spatial explanation. The downstream
system will first identify fractured levels with a future series classifier and
the existing per-vertebra MIL models, then invoke a localizer only for a crop
believed to be positive.

### Evidence

The original d32 MIL model is the strongest fully fold-excluded classifier at
0.9434 overall and 0.9350 macro five-fold AUC. Slice-student initialization
improved d32 only modestly and regressed 3x32, so it is an unnecessary and
inconclusive dependency for supervised box localization. The 7,217 native boxes
have no vertebral-level field; manufacturing C1-C7 box classes would therefore
add noisy labels that the downstream binary localization task does not need.

Exact crop geometry can be reconstructed from the OOF vertebra mask, native
slice manifest, stored crop scale, and close-square mask bounds. Projection
produced 457 usable positive vertebra crops from 234 boxed studies. One boxed
study's boxes did not intersect its only positive crop and is excluded.

### Alternatives considered

- Assigning each box to C1-C7 with the vertebra masks was rejected because it
  creates superfluous semantic classes and avoidable assignment noise.
- Initializing from the mask-free slice student was rejected for d32 because
  its extra distillation step did not conclusively improve the MIL models.
- Fine-tuning the encoder immediately was deferred to preserve the validated
  fracture representation and isolate whether a lightweight decoder can use
  its spatial features.
- Training only annotated positive slices was rejected because unboxed slices
  within an exhaustively box-annotated positive crop provide useful negative
  depth supervision.

### Decision

Use the original fold-specific d32 MIL backbone, frozen and held in evaluation
mode. Bypass the GRU, attention, slice classifier, and bag classifier. Decode
the multiscale MaxViT feature maps with one shared U-Net decoder into a single
binary fracture heatmap per depth position. Vertebral identity only selects the
input crop and never appears as a target class or model input.

Train fold 0 first with Dice plus focal loss, 96x96 targets, 32 positions,
batch 4/GPU on two GPUs without accumulation, BF16, and 12 epochs. Select on
slice AP and also report slice AUC, pointing accuracy, and positive-slice Dice.

### Consequences and follow-up

This experiment evaluates localization under oracle-positive vertebra routing;
it is not an end-to-end study result. If fold 0 shows useful spatial pointing,
run the remaining d32 folds, then implement the matched 3x32 formulation with
three binary maps per triplet. Only after that comparison should encoder
unfreezing, higher-resolution decoder outputs, or fusion of d32 and 3x32
heatmaps be considered.

Fold 0 completed with a best-AP epoch-10 checkpoint at slice AUC/AP
0.8718/0.8503, pointing accuracy 0.8180, and fixed-threshold positive-slice
Dice 0.3925. Pointing accuracy peaked earlier at 0.8340, while AP continued to
improve through epoch 10. This is sufficient quantitative evidence to retain
the frozen-encoder formulation, but heatmap overlays should be reviewed before
committing compute to all remaining folds because box masks are coarse and Dice
alone is not the intended clinical visualization criterion.

The quantitative result was accepted as sufficient to run the remaining d32
folds and a matched five-fold 3x32 localizer. The 3x32 decoder emits three
binary maps per grouped position so localization retains all 96 resampled
slices; the three channels represent relative slice position within a triplet,
not anatomy. This provides a direct test of whether the wider depth sampling
improves slice discrimination or spatial pointing over d32.

All folds completed. At each fold's AP-selected checkpoint, d32 averaged
slice AUC/AP 0.8567/0.8359, pointing accuracy 0.8241, and positive Dice
0.4285. The 3x32 averages were 0.8775/0.7194, 0.8094, and 0.4312. The apparent
AP regression is confounded by evaluation-grid prevalence: 3x32 predicts 96
slice positions while d32 predicts 32, so it introduces many more negative
positions. AUC favors 3x32, pointing modestly favors d32, and thresholded Dice
is tied. Do not select a winner from these internal-grid metrics. The next
comparison must map both outputs to native DICOM slices, evaluate against the
same native box targets, and then measure a d32-plus-3x32 ensemble.

The native-grid evaluation used fold-best EMA weights and equal probability
averaging on a common 96-position crop grid. D32 was the stronger single model
at mean-fold native AUC/AP 0.8861/0.8509 versus 0.8483/0.7881 for 3x32. The
ensemble reached 0.8900/0.8538, and its larger improvement was in localization
behavior: within-study AUC rose from d32's 0.8727 to 0.8829, pointing accuracy
from 0.8448 to 0.8656, positive Dice from 0.4319 to 0.4382, and top-1 study
recall from 0.8803 to 0.9060. Select the equal-probability ensemble for
downstream visualization while retaining d32 as the efficient single-model
fallback. These remain oracle-positive-vertebra results and must not be
presented as end-to-end series performance.

## 2026-08-09: Build the eight-output series stage from seven MIL tokens

### Context

The per-vertebra classifiers do not yet jointly predict the competition's
overall fracture target and C1-C7 targets. A second stage must contextualize
the seven crop predictions without introducing acquisition metadata or
changing the natural study distribution.

### Evidence

The original d32 and 3x32 MaxViT MIL models provide a 512-dimensional
attention-pooled post-BiGRU feature for each vertebra, plus bag and MIL logits
and attention statistics. The study manifest contains 2,019 series on the
established five study-level folds. The native crop manifest is missing seven
of the expected 14,133 study-level rows, so missingness must be explicit rather
than silently dropping a series. EMA weights generated the upstream validation
metrics and are therefore the appropriate frozen extractors.

The competition loss is a weighted binary log loss after flattening each
eight-output series row: negative/positive vertebral predictions have weights
1/2, while negative/positive overall predictions have weights 7/14. The exact
validation metric must be computed after globally concatenating predictions;
averaging separately normalized batch ratios is not identical.

### Alternatives considered

- A flat 224-token model over all depth positions was deferred until the
  seven-token baseline establishes whether additional depth detail is useful.
- Mixing OOF features from five different upstream embedding spaces in one
  second-stage training fold was rejected. For outer fold `f`, both its
  second-stage training and validation series are extracted by upstream fold
  `f`; only the outer validation series were unseen by that extractor.
- Early d32/3x32 feature concatenation was deferred until separate family
  models and a late prediction ensemble are measured.
- Weighted sampling was rejected. Studies are shuffled uniformly in their
  natural prevalence; competition weights appear only in the loss.

### Decision

Create one ordered token per vertebra from the 512-dimensional pooled MIL
feature, bag logit, normalized LSE/MIL logit, normalized attention entropy, and
maximum attention (516 values total). Replace a missing crop with a learned
token, add learned C1-C7 positional embeddings, and contextualize the seven
tokens with a one-layer bidirectional GRU. Apply a shared per-level head to the
seven contextual tokens and a separate attention-pooled overall head, emitting
`overall,C1,...,C7` in that exact order.

Test paired configurations with and without the proposed LayerNorm-plus-linear
projection to 256 dimensions. Train on the natural study distribution with the
exact competition-weighted loss, checkpoint on globally aggregated validation
competition loss, and report overall, per-level, and macro level AUC. Start
with d32, repeat for 3x32, then evaluate a late ensemble.

### Consequences and follow-up

Fold-specific feature caches are required before the lightweight series models
can train. Extraction uses frozen EMA checkpoints, disables training-only
activation checkpointing, and is resumable by output file. A 256-slice MaxViT
inference chunk exceeded PyTorch's 32-bit tensor indexing limit; use the tested
128-slice chunk. This is an infrastructure constraint, not model evidence.
All feature caches and series experiment artifacts must live under
`/mnt/champaca`, which has more than 2 TiB free; the root filesystem has only
about 8 GiB free. Series configs therefore default to
`/mnt/champaca/experiments/cspine-fx` and explicitly use the shared HTTP MLflow
server rather than inheriting a workstation-local SQLite URI.

## 2026-08-13: Package the pipeline as two versioned Hugging Face models

### Context

The selected experiments now cover vertebra segmentation, crop-level feature
extraction, eight-output series prediction, and conditional fracture
localization. They need one reproducible inference contract without coupling a
large segmentation ensemble to the already large fracture artifact.

### Evidence

The segmentation student reaches 0.8037 mean re-evaluated EMA Dice and is sufficient
for crop construction. The selected equal-probability ensemble of the four raw
series architectures reaches 0.25981 OOF competition loss, 0.93930 overall AUC,
and 0.94087 macro C1-C7 AUC. The d32/3x32 localization ensemble is stronger than
either localization family alone on the boxed positive-vertebra evaluation.
Direct parity testing on a real study showed that release-time d32 and 3x32 crop
tensors are exactly equal to the tensors from the training dataset. DICOM
loading also exactly reproduced the cached windowed PNG volume and slice order.

### Alternatives considered

- One monolithic repository was rejected because segmentation is independently
  reusable and would make the already 2.6 GB prediction package larger.
- Shipping Lightning checkpoints was rejected because they contain optimizer
  and training state and require unsafe pickle deserialization.
- Embedding TotalSegmentator was rejected because it is much slower and the
  trained student is adequate for cropping.
- Uploading artifacts immediately was rejected because remote repository IDs
  and publication authorization have not been provided.

### Decision

Release two typed, versioned model directories. `cspine-segmentation` contains
the five C1-C7 SafeTensors checkpoints. `cspine-fracture-classifier` contains
five folds of both MIL families, all four raw series heads, and both binary
localization decoders. A canonical Python API and CLI accept one physically
ordered DICOM series, reproduce center-400/width-1800 uint8 windowing, run the
two stages, and return overall/C1-C7 probabilities plus conditional heatmaps.
Localization files retain exact slice-index, instance-number, and crop geometry;
the API can lazily restore each heatmap to the native axial coordinate system.

The model directories bundle a project wheel, model card, machine-readable
manifest, checksums, sanitized provenance, requirements, and only SafeTensors
weights. Local paths and Hugging Face model IDs use the same `from_pretrained`
entry points. Inference loads folds sequentially and processes crops one at a
time by default to bound GPU memory.

### Consequences and follow-up

The staged packages live under `/mnt/champaca/huggingface/cspine`, not the
primary workspace filesystem. They remain research-only and unpublished.
All SafeTensors and representative strict model loads must pass before release.
Full all-fold GPU inference should be timed when the shared GPUs are free;
thick-slice robustness and external-domain validation remain open work.

## 2026-11-08: Define the public final-v1 reproduction and release contract

### Context

The project is being prepared for publication at
`i-pan/ct-cspine-fracture-detection`. A deep publication review found that the
research worktree did not yet encode the exact selected pipeline in one
portable contract: documentation stopped after MIL training, several generated
manifests had no CLI, release paths were workstation-specific, and segmentation
validation used EMA while export used raw weights.

### Evidence

The selected four-head series ensemble remains the best measured OOF model at
0.25981 competition loss, 0.93930 overall AUC, and 0.94087 macro C1-C7 AUC.
Its upstream models are the original d32 and 3x32 MIL families, not the
slice-initialized d32 ablation. The original d32 models are also the encoders
used by the selected localization and series stages. The deterministic teacher
QC rule that rejects only an all-background C1-C7 teacher result exactly
reproduces the existing cache: 1,931 of 1,932 pseudo-labeled studies, excluding
study `1.2.826.0.1.3680043.20574`.

Segmentation checkpoints contain distinct raw and EMA states. Validation swaps
in EMA weights, so exporting raw weights does not reproduce the state used for
checkpoint selection. A matched five-fold evaluation found 0.80367 mean Dice
for EMA and 0.80315 for raw weights; EMA won four folds. This evaluation also
identified and corrected the older 0.8078 documentation summary, which did not
match the selected checkpoint logs (0.80328 mean).

### Alternatives considered

- The two-head reduced-compute series ensemble was rejected as the primary
  release because the four-head ensemble has the best competition loss and
  overall reported result. It remains a documented deployment alternative.
- Publishing only selected configs was rejected; ablation configs and
  summarized results will remain available, but outside the authoritative
  release path.
- Retaining raw segmentation export was rejected unless a later matched
  evaluation demonstrates that it is preferable. EMA is the default because
  it is the evaluated checkpoint state.
- Hard-coded local run roots were rejected in favor of explicit artifact roots
  and a versioned release manifest.

### Decision

Publish the complete code under Apache-2.0 and define `release/final-v1.json`
as the machine-readable source of truth. The final classifier uses equal
probability averaging of four raw/no-projection series heads: d32 7x1, d32
7x32, 3x32 7x1, and 3x32 7x32. Both original MIL families and both binary
localizers are included. The special loader-fixed 3x32 fold-4 checkpoint is
recorded explicitly. Segmentation, MIL, and localizer weights use EMA; series
heads use their model state.

The public repository will include selected and ablation configs plus
summarized experiment results. Data, processed images, manifests containing
local paths, checkpoints, and experiment directories remain excluded from
Git. Users must obtain the RSNA/Kaggle data themselves.

### Consequences and follow-up

The GitHub remote is configured as `origin`, while SKPv2 remains `template`.
The remote's independent Apache-license commit must be integrated without
discarding either history. Before `final-v1` is promoted from candidate to
released, populate its source revision and artifact hashes, reproduce the OOF
ensemble through public code, validate EMA segmentation metrics, test both HF
packages in a clean environment, and complete a clean-clone reproduction gate.

The original training runs did not consistently specify seeds, so the release
will distinguish exact published-artifact hashes from retraining
reproducibility within documented metric tolerances.

Subsequent implementation reproduced the selected four-head OOF metrics from
the manifest-resolved checkpoints and strict-loaded all 45 newly exported
artifacts. The release path now fails on stale export outputs, verifies
SafeTensor hashes at load time, and uses a pinned local `pytorchvideo` package
whose X3D initialization is byte-identical to the previously cached Torch Hub
state. These results support the manifest as the authoritative selection
contract. Functional source revision `3d9d8946d6b86692d372efbb3630099c7db49611`
is now recorded; final HF repository IDs remain intentionally unset.

## Decision-entry template

### YYYY-MM-DD: Decision title

#### Context

What problem or uncertainty prompted the decision?

#### Evidence

What measurements, qualitative review, failure analysis, or external references
informed it? Clearly distinguish preliminary, cross-validation, holdout, and
external-test evidence.

#### Alternatives considered

What other viable options were considered, and what were their tradeoffs?

#### Decision

What was selected, rejected, deferred, or left conditional?

#### Consequences and follow-up

What changes in the training, evaluation, inference, documentation, or release
contract? What still needs to be tested?


## 2026-11-08: Checkpoint cleanup proposal, not executed

Proposed `docs/cleanup/cspine-cleanup-20261108-v1/proposal.md`: remove 209 unused/source weight files and 93 aliases (39.180 GiB); keep 45 staged inference files and hold only five selected segmentation source checkpoints plus aliases. Staged segmentation uses raw state while final-v1 requires EMA, so those sources are required for a corrected export. Approximate reproduction is supported by retained effective MLflow settings, configs, source snapshot and logs; exact resume and historical initialization/data availability are not guaranteed. Manifest SHA-256: `3fb8a675e6e95f899464b0ca48519c7c82c18c6f3a76b58417acd8c99d3da116`. No cleanup approval has been received, no experiment artifacts changed, and no training/export launched.


## 2026-11-08: Cleanup reinspection supersedes v1

The new series OOF evaluator is running and consumes twenty selected series-head checkpoints proposed for removal in v1. V2 moves those checkpoints and twenty aliases to hold, in addition to five segmentation source checkpoints/aliases held for EMA re-export. Revised proposal: `docs/cleanup/cspine-cleanup-20261108-v2/proposal.md`. No deletion performed.
