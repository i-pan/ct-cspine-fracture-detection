<!-- skpv2-project-documentation: experiment-history -->

# Experiment History

This is the project's experiment registry. Keep it current as experiments are
launched, stopped, fail, or become selected baselines.

This document is complementary to [DECISION_LOG.md](DECISION_LOG.md), and some
overlap is expected:

- **Experiment History** emphasizes what was run, its configuration, its result,
  and its disposition.
- **Decision Log** contains the deeper analysis explaining why an experiment or
  direction was selected, rejected, revised, or deferred.

Template-framework decisions belong in the repository-root
[`DECISIONS.md`](../DECISIONS.md), not here.

## Selected results

| Experiment | Configuration | Result | Decision |
|---|---|---|---|
| C1-C7 segmentation baseline | `cspine_segmentation_x3d_s`, 128-cube, five folds | Dice 0.6745 mean across fold-best checkpoints | Retain as manual-only baseline |
| TotalSegmentator teacher benchmark | v2.16.0 `total`, 1.5 mm, resampling order 3 | Dice 0.8940 +/- 0.0170 on all 87 manual masks | Generate compact pseudo-label cache |
| Pseudo-label segmentation student | TotalSegmentator pretraining plus manual fine-tuning, five folds | Re-evaluated mean EMA Dice 0.8037; raw 0.8031 | Sufficiently fast crop localizer; teacher remains the quality ceiling |
| Fracture MIL, adjacent-slice depth 32 | Image-only close-square MaxViT-Tiny, five folds | Mean macro AUC 0.9350 +/- 0.0126; overall AUC 0.9434 +/- 0.0094 | Selected primary classifier |
| Fracture MIL, 96 to 3x32 stack | Image-only close-square MaxViT-Tiny, five folds | Mean macro AUC 0.9299 +/- 0.0100; overall AUC 0.9393 +/- 0.0083 | Retain as complementary slice-localization model |
| Weak slice localization | OOF depth-32 plus 3x32 native-grid ensemble on 235 boxed studies | Mean fold AUC 0.8905; AP 0.7366; within-study AUC 0.8688 | Strong pseudolabel signal; ensemble selected |
| Mask-free 2.5D slice student | MaxViT-Tiny 384, adjacent three-slice channels, soft OOF ensemble targets | Epoch 0 mean-fold AUC/AP 0.9118/0.7802; pooled 0.9104/0.7770 | Student selected for slice localization and MIL encoder initialization |
| Slice-initialized d32 MIL | Slice-student MaxViT encoder, 2 epochs, 10% warmup plus cosine 2e-4 to 2e-6, five folds | Overall/macro AUC 0.9450 +/- 0.0090 / 0.9382 +/- 0.0098 versus original 0.9434/0.9350 | Retain transfer for d32; modest gain and much shorter training |
| Slice-initialized 3x32 MIL | Same transfer and schedule, loader-fixed 3x32, five folds | Overall/macro AUC 0.9372 +/- 0.0076 / 0.9277 +/- 0.0098 versus original 0.9393/0.9299 | Reject transfer for 3x32 |
| Binary fracture heatmaps | Frozen original d32 MIL MaxViT encoder plus shared 2D U-Net decoder, fold 0 | Best AP checkpoint epoch 10: slice AUC/AP 0.8718/0.8503; pointing accuracy 0.8180; positive Dice 0.3925 | Promising oracle-positive localization; proceed to qualitative review and remaining folds |
| Binary fracture heatmaps, d32 CV | Frozen original fold-specific d32 encoders plus binary decoder, five folds | Mean fold-best AUC/AP 0.8567/0.8359; pointing 0.8241; Dice 0.4285 | Retain; compare on native grid before selecting/ensembling |
| Binary fracture heatmaps, 3x32 CV | Frozen original fold-specific 3x32 encoders plus three-map decoder, five folds | Mean fold-best AUC/AP 0.8775/0.7194; pointing 0.8094; Dice 0.4312 | Higher fine-grid AUC; AP not directly comparable to d32 because prevalence differs |
| Localization heatmap ensemble | OOF EMA d32 and 3x32 probabilities averaged on a common 96-position grid, then mapped to native slices | Mean-fold native AUC/AP 0.8900/0.8538; within-study AUC 0.8829; pointing 0.8656 | Select ensemble for visualization/localization |
| Seven-token series-head ensemble | Five-fold OOF equal-probability average of raw d32 and raw 3x32 heads | Competition loss 0.26594; overall/level-macro AUC 0.93914/0.93833 | Select over either single head; projection variants do not improve the ensemble |
| Combined series-head ensemble | Equal average of the selected 7x1 and 7x32 raw-family OOF ensembles | Competition loss 0.25981; overall/level-macro AUC 0.93930/0.94087 | Selected series-stage ensemble; depth tokens add complementary level signal |
| Two-model series-head ensemble | Equal-probability 7x1-3x32 plus 7x32-d32 raw heads, selected from all six pairs | Competition loss 0.26057; overall/level-macro AUC 0.93897/0.93851 | Preferred reduced-compute pair; retains nearly all four-model loss gain |
| End-to-end release artifacts | Two typed HF-ready packages; 5-fold segmenter plus full four-head fracture/localization ensemble | 45/45 SafeTensors validated; strict representative component loads passed; real-study DICOM and crop preprocessing parity exact | Staged under `/mnt/champaca`; publication awaits explicit repository authorization |

## What the experiments have established

All fracture-classification values below are preliminary fold-0 results unless
explicitly described as cross-validated. The current five-fold runs are needed
before treating small differences as stable.

- **Localization is good enough to proceed.** The manual-only X3D-S segmenter
  reached mean Dice 0.6745. Full-resolution TotalSegmentator reached 0.8940 on
  the same 87 manual cases. A fold-safe student pretrained on 1,931 compact
  pseudo-labels and then fine-tuned manually reached 0.8037 mean EMA Dice. Manual fine-tuning
  improved the pseudo-pretrained fold mean only modestly (approximately 0.009),
  while the student is much faster than the teacher. Precise boundaries are not
  required for the current use: crop localization and a soft target-level mask.
- **Weighted sampling is important for the weak vertebra labels.** For the
  matched EfficientNetV2-S depth-32 pipeline, weighted sampling reached macro
  AUC 0.8900-0.8914 versus 0.8433 with the natural distribution. BCE itself
  remained unweighted. Positive repetition can overfit after the early peak,
  so eight epochs and best-checkpoint selection are adequate.
- **Larger batches did not help.** Increasing EfficientNetV2-S from 2 to 4
  crops/GPU reduced macro AUC from 0.8900 to 0.8706. The project therefore
  retains 2/GPU, global batch 4, without gradient accumulation.
- **More depth was not automatically better.** At matched EfficientNetV2-S
  settings, fixed depth 32 reached 0.8900 macro AUC; variable depth 48 reached
  0.8858, variable depth 64 reached 0.8854, and the uniformly spaced 16x5 stack
  reached 0.8742. These results favored depth 32, although the later MaxViT
  96-to-3x32 representation is competitive enough to cross-validate.
- **Backbone and crop geometry produced the largest classification gains.** A
  standard 384-pixel MaxViT-Tiny improved the EfficientNetV2-S macro AUC from
  0.8900 to 0.9080. Tight level-mask depth bounds plus a centered square with
  only 5% in-plane padding then raised image-only MaxViT to 0.9328. The square
  avoids aspect-ratio distortion and makes the vertebra occupy substantially
  more of the 384-pixel field of view.
- **The current classifier is image-only.** It receives previous/current/next
  windowed CT slices and the target vertebra's soft segmentation mask. Source
  dimensions, native depth, slice spacing, and C1-C7 identity do not enter the
  model. The earlier metadata-enabled close-square run reached 0.9299 macro and
  0.9392 overall AUC; removing all non-image inputs improved this to 0.9328 and
  0.9439. Metadata therefore does not explain the high score. The planned
  C1-C7 embedding ablation was cancelled.
- **Depth 32 and 3x32 currently trade metrics.** Adjacent-slice depth 32 leads
  macro AUC, 0.9328 versus 0.9312. Resampling to 96 slices and grouping
  non-overlapping triplets leads overall AUC, 0.9475 versus 0.9439. The
  differences are small enough that both are being trained on folds 1-4 and
  may be complementary in an ensemble.
- **The first depth-preserving 3D MIL designs underperformed.** X3D-S, X3D-L,
  and CSN-26 with H/W pooling, per-depth logits, and normalized LSE reached
  macro AUCs 0.7788, 0.7739, and 0.7258. Replacing slice LSE with global D/H/W
  pooling improved X3D-S to 0.8409. Closely reproducing the original grayscale
  X3D-L pipeline was much stronger: 0.8995 macro and 0.9159 overall AUC on the
  current fold. The full preprocessing, sampling, pooling, and optimization
  contract mattered more than merely selecting a 3D backbone.
- **The 512-pixel MaxViT attempt is inconclusive.** It was stopped while using
  a MaxViT variant pretrained for 384-pixel inputs, so its poor early result is
  not evidence that higher axial resolution is harmful. A properly matched
  MaxViT-512 remains a possible later experiment; 256 pixels is not currently
  justified by the observed benefit from tighter, larger vertebra views.
- **MIL logits contain useful slice-level information.** On the boxed fold-0
  subset, mapped native-slice logits achieved AUC 0.8647, AP 0.6639, and 100%
  box-slice crop coverage. This supports training a 2.5D slice classifier on
  high-confidence pseudolabels before using its encoder to initialize MIL, but
  pseudolabel thresholds and leakage-safe evaluation remain outstanding.

## Latest completed milestone

- 2026-08-09: Evaluated the five-fold OOF d32, 3x32, and equal-probability
  heatmap ensemble. D32 maps were trilinearly expanded from 32 to the common
  96-position crop grid; 3x32's three output channels were restored to their
  original consecutive-slice order. Probabilities were averaged before taking
  spatial maxima and mapping depth scores back to native DICOM slices. All
  inference used the EMA state stored in each AP-selected checkpoint.

  Mean-fold native max-over-crop slice AUC/AP was 0.8861/0.8509 for d32,
  0.8483/0.7881 for 3x32, and 0.8900/0.8538 for the ensemble. Mean within-study
  AUC was 0.8727, 0.8381, and 0.8829, respectively. The ensemble also improved
  common-grid pointing accuracy from 0.8448 (d32) and 0.8101 (3x32) to 0.8656,
  and positive-map Dice from approximately 0.4320 for either single model to
  0.4382. Native top-1 study recall improved from 0.8803 for d32 to 0.9060 for
  the ensemble. It covered 7,151 of 7,215 annotated slices among the 234
  geometrically routable boxed studies. Pooled OOF ensemble native AUC/AP was
  0.8893/0.8519. Select the ensemble for downstream heatmaps; retain d32 as the
  stronger single model. Artifacts and the full per-fold report are under
  `evaluations/localization_decoder_ensemble_cv` in the processed-data root.

- 2026-08-09: Completed the remaining localization cross-validation queue.
  The first stage trained d32 folds 1-4 under the exact selected fold-0
  contract and original fold-specific MIL encoders. The second stage trained
  3x32 folds 0-4. For 3x32, each of 32 grouped encoder inputs predicts three
  independent binary 96x96 maps corresponding to its three constituent CT
  slices; these are slice outputs, not C1-C7 classes. The 3x32 fold-4 localizer
  uses the loader-fixed classification encoder checkpoint. Run IDs are
  `cspine-localize-d32-frozen-mil-binary-v1` and
  `cspine-localize-3x32-frozen-mil-binary-v1`.

  D32 fold-best epochs were 10, 11, 9, 5, and 11. Mean fold-best slice
  AUC/AP, pointing accuracy, and positive Dice were 0.8567, 0.8359, 0.8241,
  and 0.4285. The corresponding 3x32 values, at epochs 10, 11, 10, 11, and
  11, were 0.8775, 0.7194, 0.8094, and 0.4312. The 3x32 grid contains 96
  positions versus 32 for d32, producing substantially more negative model
  positions. Therefore AP is not directly comparable across the two grids;
  AUC favors 3x32 by 0.0208, Dice is essentially tied, and pointing accuracy
  favors d32 by 0.0147. Export both models to native DICOM slices before model
  selection or measuring an ensemble.

- 2026-08-08: Completed fold 0 of
  `cspine-localize-d32-frozen-mil-binary-v1`. The dataset contains 457
  geometrically routed positive vertebra crops from 234 boxed studies; folds
  contain 90-94 crops each. Native boxes are clipped to each crop and projected
  through the exact close-square d32 transform. This yields 10,030 routed box
  instances at 6,243 positive model positions; overlap can intentionally route
  one native box into multiple positive crops. The one boxed study omitted has
  a positive C6 label but no box intersecting its predicted C6 crop, so it is
  excluded rather than given an incorrect negative localization target.

  The original fold-specific d32 MIL MaxViT encoder is loaded and completely
  frozen; the GRU, attention, bag/MIL heads, level identity, and metadata are
  absent. A shared 3.1M-parameter U-Net decoder predicts one binary 96x96 map
  for each of 32 positions. Training uses Dice plus focal loss, batch 4/GPU,
  BF16 two-GPU DDP, no accumulation, 12 epochs, and selects on validation
  slice AP. Validation also reports slice AUC, pointing accuracy, and positive-
  slice Dice. Run and config:
  `cspine_fracture_localization_d32/cspine-localize-d32-frozen-mil-binary-v1`.
  The selected epoch-10 checkpoint reached slice AUC/AP 0.8718/0.8503,
  pointing accuracy 0.8180, and fixed-threshold positive-slice Dice 0.3925.
  Slice AUC peaked at 0.8719 at epoch 9, pointing accuracy peaked at 0.8340
  at epoch 2, and positive-slice Dice peaked at 0.3939 at epoch 11. The AP
  curve improved from 0.8189 at epoch 0 to 0.8503 at epoch 10 and then changed
  only slightly, so the 12-epoch horizon was adequate. The best checkpoint is
  `fold0/checkpoints/epoch=010-val_metric=0.8503.ckpt`.

- 2026-08-05: Run `cspine-vertebra-maxvitt-mil-d32-close-square-imageonly-bs2-v1`
  completed all five folds at mean best-checkpoint AUC 0.9434 overall and
  0.9350 macro. Run
  `cspine-vertebra-maxvitt-mil-d32-stack3x96-imageonly-bs2-v1`, using the
  loader-fixed fold-4 retry, completed at 0.9393 overall and 0.9299 macro.
  Depth 32 leads by 0.0041 overall and 0.0052 macro AUC. The original fold 4
  was stopped after epoch 2, whose
  checkpoint reached 0.9203 overall and 0.9105 macro AUC. Runtime 96-slice
  interpolation plus persistent-worker prefetching caused progressive CPU
  allocation and shared-memory-copy overhead: throughput fell from 1.10 to
  0.32 steps/s while one DDP rank waited. Fold 4 was relaunched from scratch as
  `cspine-vertebra-maxvitt-mil-d32-stack3x96-imageonly-fold4-loaderfix-v1`
  with two workers/rank, non-persistent workers, and Torch/OMP-family thread
  pools limited to one; throughput remained 1.10 steps/s and fold 4 reached
  0.9423 overall and 0.9384 macro AUC at epoch 5.

  Five-fold out-of-fold slice localization then evaluated all 235 studies with
  fracture boxes. Mean per-fold max-over-level slice AUC/AP were
  0.8688/0.6902 for depth 32, 0.8644/0.6833 for 3x32, and 0.8905/0.7366 for
  their native-grid logit ensemble. The ensemble improved both AUC and AP on
  every fold. Its mean within-study AUC was 0.8688; top-1/3/5/10 study recall
  was 0.8255/0.8723/0.8894/0.9191. It covered 7,198 of 7,217 annotated slices
  (99.74%). Raw pooled OOF AUC/AP were 0.8754/0.7232; these are lower than the
  mean fold metrics because independently trained folds have different logit
  scales. Artifacts and the complete per-fold report are stored under
  `evaluations/slice_localization_cv` in the processed-data root.

- 2026-08-05: Generated 448,485 native-slice OOF soft targets from the d32 and
  3x32 ensemble and trained one mask-free MaxViT-Tiny 384 classifier with the
  previous/current/next PNGs as three input channels. Run
  `cspine-slice-2dc-maxvitt-384-soft-oof-ensemble-bs32-nogc-v1` peaked after
  its first full epoch: mean-fold manual slice AUC/AP 0.9118/0.7802 and pooled
  AUC/AP 0.9104/0.7770. The corresponding teacher values were 0.8905/0.7366
  mean-fold and 0.8754/0.7232 pooled. The student improved AUC and AP in every
  fold. Mean-fold AUC declined to 0.9062 at epochs 1-2 and approximately
  0.903 thereafter, indicating that one full pseudolabel pass was sufficient.
  Manual box labels were used only for validation, while the single student
  did see all images with OOF teacher targets; this is a full-data distillation
  estimate, not fold-excluded student CV.

- 2026-08-07: Fold-0 d32 MIL initialized from the selected slice encoder
  completed two epochs with 10% linear warmup and cosine LR from a 2e-4 peak
  to 2e-6. Overall/macro AUC was 0.9314/0.9221 at epoch 0 and
  **0.9473/0.9359** at epoch 1. The original ImageNet-initialized run reached
  0.8970/0.8757 and 0.9303/0.9182 at the matched epochs, and its eight-epoch
  best was 0.9439/0.9328. Thus transfer gained 0.0034 overall and 0.0031 macro
  AUC over the prior best while reducing the horizon from eight epochs to two.
  The slice student trained on OOF targets for every study, including this
  validation fold, so the result retains the documented transductive-init
  caveat.

  The completed five-fold d32 transfer result was 0.9450 +/- 0.0090 overall
  and 0.9382 +/- 0.0098 macro AUC, gains of 0.0016 and 0.0031 over the original
  eight-epoch means. Macro AUC improved on three of five folds, so the benefit
  is modest rather than universal, but it was achieved with one quarter of the
  training horizon. The identical five-fold 3x32 transfer run reached
  0.9372/0.9277, decreasing overall/macro AUC by 0.0021/0.0022 from its
  original result; slice initialization is therefore retained only for d32.

## Iteration log

The entries below preserve the chronological experiment and queue history;
they are not all still active.

- 2026-08-01: Implemented and queued the compact fixed-volume 3D Halton search
  in the shared `cspine_vertebra_fracture_classification` MLflow experiment.
  The completed/current X3D-S `64x288x288`, batch-2/GPU global-pooling run is
  the anchor; its macro AUC trajectory reached 0.8409 after epoch 6. Five
  deterministic Halton points cover X3D-S, X3D-L, CSN-26, voxel-matched
  `48x320x320`, `64x288x288`, and `80x256x256` inputs, batch 2/4 per GPU,
  8/12/16 epochs, and log-uniform LR 1e-4 to 4e-4. Each trial gets a two-batch
  memory smoke test. Batch-4 OOMs retry at batch 2 without accumulation.
  Matched-epoch pruning uses a 0.05 macro-AUC margin after epoch 4 and 0.03
  after epoch 8. Scheduler horizon matches each sampled training horizon with
  one warm-up epoch. The queue starts automatically after the anchor exits.

- 2026-07-29: Full-resolution TotalSegmentator pseudo-label generation
  completed for all 1,932 studies without manual masks. One scan contained no
  predicted C1-C7 voxels and was excluded; 1,931 QC-passing 128-cube samples
  remain for fold-safe pretraining.
- 2026-07-29: EfficientNetV2-S 2.5D vertebra MIL fold-0 batch-size comparison.
  `cspine-vertebra-mil-effv2s-v1` is training at per-GPU/global batch 2/4.
  Run `cspine-vertebra-mil-effv2s-bs4-v1` is queued at per-GPU/global batch
  4/8. It uses 20 epochs, LR 2e-4, the same weighted sampler and loss, BF16
  DDP, no gradient accumulation, and a 64-slice encoder chunk. Run the
  per-GPU/global batch 8/16 variant only if 4/8 improves or meaningfully
  matches the baseline. Short throughput probes measured approximately 7.6,
  6.4, and 5.1 crops/s for global batches 4, 8, and 16; these are efficiency
  measurements, not model-quality results.
- 2026-07-29: The weighted-with-replacement v1 run peaked at 0.8914 fold-0
  macro level AUC and then declined. Its 11,305-sample balanced epoch produced
  about 5,652 positive draws from only 1,155 positive training crops, or about
  4.9 draws per positive per epoch. Stop v1 after retaining its best
  checkpoint. The revised baseline removes artificial sampling and presents
  each crop once per shuffled epoch at the natural 1,155/11,305 (10.2%)
  positive prevalence. BCE remains unweighted so sampling is the only changed
  training variable.
- 2026-07-29: Restarted the natural-distribution depth-32 baseline with a
  12-epoch horizon. After it completes, generate a separate variable-depth
  crop cache directly from native PNG volumes and train
  `cspine_vertebra_mil_effnetv2_s_d64` for the same 12 epochs at batch 2/GPU.
  Preserve every native slice for depths up to 64 and downsample only longer
  crops. Dynamic collation right-pads within each batch, but padded frames
  bypass the CNN and are excluded from a packed BiGRU, attention, and LSE
  pooling. The run retains the same backbone, LR, loss, sampling, augmentation,
  axial resolution, encoder chunk size, and fold.
- 2026-07-30: Natural-distribution depth-32 run
  `cspine-vertebra-mil-effv2s-natural-v2` completed 12 epochs. Best fold-0
  macro level AUC was 0.8433 at epoch 9 (overall AUC 0.8659), versus 0.8914
  for the weighted-with-replacement v1. The final macro AUC was 0.8395.
  Variable-depth crop generation then completed all 14,126 samples (45 GiB);
  `cspine-vertebra-mil-effv2s-natural-d64-variable-v1` is training.
- 2026-07-30: Stopped the natural-sampling variable-depth run during epoch 1.
  Restore `WeightedSampler` and run a matched 12-epoch comparison at fixed
  depth 32 followed by native depth capped at 64. Both use unweighted BCE,
  batch 2/GPU, the same LR schedule, fold, and all other model settings.
- 2026-07-30: Inserted
  `cspine_vertebra_mil_effnetv2_s_d16_stack5` between the weighted depth-32
  and variable-depth-64 runs. Build a full native-depth crop cache, select 80
  nearest native slice indices uniformly across each crop, and reshape them
  into 16 groups of five CT channels. An exactly 80-slice crop uses every
  slice exactly once. Crops shorter than 80 repeat indices, so neighboring
  groups may overlap in the source volume; crops longer than 80 skip indices
  uniformly. The sixth channel is the target vertebra mask at the middle slice
  of each five-slice group.
- 2026-07-30: Reduced the two remaining ablations to 8 training epochs. Keep
  the scheduler horizon at 12 epochs so their learning-rate values through
  epoch 7 match the completed depth-32 comparator; only the unnecessary
  post-peak epochs are omitted.
- 2026-07-30: Weighted 12-epoch depth-32 run
  `cspine-vertebra-mil-effv2s-weighted12-d32-v2` completed. Best fold-0
  macro level AUC was 0.8900 at epoch 6 (overall AUC 0.9028), effectively
  matching the longer weighted v1 result of 0.8914. Final macro AUC was
  0.8806. Native-depth cache generation is in progress before the 16x5 run.
- 2026-07-30: Weighted 16x5 run
  `cspine-vertebra-mil-effv2s-weighted8-d16-stack5-v1` completed. Best
  fold-0 macro level AUC was 0.8742 at epoch 6 (overall AUC 0.8889);
  the final macro AUC was 0.8647. The first weighted variable-depth d64
  launch failed at 68% of epoch 0 when its ranks entered different NCCL
  collectives. Increase the d64 encoder chunk size from 64 to 128 so every
  local batch (at most 2 x 64 valid frames) uses one backbone invocation,
  eliminating rank-dependent chunk counts before relaunching.
- 2026-07-30: Queue depth-32 batch-size ablation
  `cspine-vertebra-mil-effv2s-weighted8-d32-bs4-v1` after the active d64
  run. It uses batch 4/GPU (global 8), weighted sampling, unweighted BCE,
  no gradient accumulation, eight training epochs, and the same 12-epoch
  scheduler trajectory. All other data, model, fold, and optimization
  settings match the selected depth-32 baseline.
- 2026-07-30: Insert a weighted variable-depth-48 run at batch 2/GPU before
  the batch-size ablation. Generate a dedicated cache capped at 48 native
  slices and compare its best fold-0 macro level AUC against the depth-32
  best of 0.8900. If d48 exceeds 0.8900, run d48 at 4/GPU next; otherwise,
  retain the queued d32 4/GPU ablation. Both conditional batch-size runs use
  weighted sampling, unweighted BCE, no accumulation, eight epochs, and the
  12-epoch reference scheduler.
- 2026-07-31: Launch the EfficientNetV2-M backbone ablation at fixed depth 32
  and batch 2/GPU. Use ImageNet-21k-pretrained then ImageNet-1k-finetuned
  weights, weighted sampling, unweighted BCE, 384-pixel axial resolution,
  no gradient accumulation, eight epochs, and the same 12-epoch reference
  scheduler as the selected EfficientNetV2-S baseline.
- 2026-07-31: Queue a MaxViT-Tiny backbone ablation after EfficientNetV2-M.
  Use the ImageNet-1k-pretrained 384-pixel MaxViT variant at fixed depth 32,
  batch 2/GPU, weighted sampling, unweighted BCE, no accumulation, eight
  epochs, and the same 12-epoch reference scheduler.
- 2026-07-31: Stop EfficientNetV2-M near the start of epoch 4 after its
  epoch-3 macro level AUC reached only 0.8299, versus 0.8534 for the matched
  EfficientNetV2-S trajectory. Retain its epoch-3 checkpoint and advance the
  queue directly to MaxViT-Tiny.
- 2026-07-31: Queue a 512x512 MaxViT-Tiny ablation after the active 384x384
  run. Generate a dedicated fixed-depth-32 crop cache from the original PNG
  volumes so the experiment retains native axial detail instead of upscaling
  384-pixel crops. Use the ImageNet-1k-pretrained 512-pixel MaxViT variant,
  batch 2/GPU, weighted sampling, unweighted BCE, no accumulation, eight
  epochs, and the same 12-epoch reference scheduler.
- 2026-07-31: Stop the 512x512 MaxViT run during epoch 2 and evaluate the
  best 384x384 checkpoint's weakly supervised slice logits on the 47 fold-0
  studies with fracture boxes. Interpolate the 32 crop positions back to the
  native DICOM slice grid and take the maximum logit across overlapping C1-C7
  crops. Across 10,468 covered slices (1,329 positive; 100% box-slice
  coverage), native-slice AUC was 0.8647, AP was 0.6639, and mean within-study
  AUC was 0.8194. Top-1/3/5/10 slice study recall was
  0.7447/0.8511/0.8511/0.8511. Mean aggregation was slightly worse (AUC
  0.8542, AP 0.6202). Retain high-confidence slice scores as candidate
  pseudolabels, but validate pseudolabel thresholds before training.
- 2026-07-31: Launch a depth-preserving X3D-S MIL comparison on the fixed
  32x384x384 crops at batch 2/GPU. Input channels are windowed CT and the
  level-specific soft mask. Set every convolutional depth stride to 1, average
  only H/W at the final encoder feature map, apply a shared linear classifier
  to all 32 depth positions, and aggregate with normalized LSE at sharpened
  temperature 0.5. Keep weighted sampling, unweighted BCE, eight epochs, the
  12-epoch scheduler trajectory, and fold 0 matched to the selected baseline.
  The first launch was stopped at 130/2,826 batches because a shell-level
  SQLite MLflow URI overrode the tracking server. Relaunch as
  `cspine-vertebra-x3ds-mil-weighted8-d32-t05-bs2-v2` through the shared
  MLflow HTTP server under experiment
  `cspine_vertebra_fracture_classification`; do not authorize local tracking.
- 2026-07-31: Queue a matched CSN-26 depth-preserving 3D MIL ablation after
  X3D-S. Use the same fixed 32x384x384 CT-plus-mask inputs, all-one temporal
  stride schedule, H/W-only pooling, 32 position logits, normalized LSE at
  temperature 0.5, batch 2/GPU, weighted sampling, unweighted BCE, fold 0,
  eight epochs, and 12-epoch scheduler trajectory. Track it as another run in
  MLflow experiment `cspine_vertebra_fracture_classification`.
- 2026-07-31: Insert X3D-L before the queued CSN-26 run. Match the active
  X3D-S contract exactly except for the larger `x3d_l` encoder: fixed
  32x384x384 CT-plus-mask inputs, all-one temporal strides, H/W-only pooling,
  normalized LSE temperature 0.5, batch 2/GPU, weighted sampling, unweighted
  BCE, fold 0, eight epochs, and the 12-epoch scheduler trajectory. Start
  CSN-26 only after X3D-L completes successfully.
- 2026-08-01: Stop CSN-26 at 33% of epoch 4; its best completed macro AUC was
  0.7258 at epoch 3. Launch X3D-S with native depth preserved up to 64 and
  uniformly downsampled only above 64. Retain the two-channel CT-plus-mask
  input, all-one temporal strides, padding-aware normalized LSE temperature
  0.5, batch 2/GPU, weighted sampling, unweighted BCE, fold 0, eight epochs,
  and the matched 12-epoch scheduler trajectory.
- 2026-08-01: Stop the variable-depth-64 X3D-S LSE run at 60% of epoch 3 and
  replace it with the 6th-place-style formulation: fixed 64x288x288 input,
  all-one temporal strides, global average pooling over the final D/H/W
  feature grid, and one linear classifier without slice logits or LSE. Retain
  CT plus the level-specific mask, batch 2/GPU, weighted sampling, unweighted
  BCE, fold 0, eight epochs, and the matched scheduler trajectory.
- 2026-08-01: Add a conditional fixed-64x288 X3D-S LSE follow-up at batch
  4/GPU. After the global-average-pooling run completes epochs 0 and 1, retain
  it only if macro AUC reaches at least 0.6858, a +0.02 improvement over the
  matched variable-depth-64 LSE epoch-1 result of 0.6658. Otherwise stop it
  and launch the LSE follow-up with no gradient accumulation, keeping all
  other fixed-64x288 data and optimization settings matched.
- 2026-08-01: Stop the compact 3D Halton queue after its first X3D-S trial
  completed and the second X3D-L trial began. Replace the search with a close
  fold-0 reproduction of the original 6th-place `chunk000` pipeline in the
  current codebase: X3D-L, grayscale `64x288x288`, tight threshold-0.4
  vertebra bounds, scipy order-0 depth resampling/repetition, scipy order-1
  axial stretch, independent Z/H/W flips, `[-1,1]` normalization, all-one
  depth strides, global D/H/W average pooling, five-sample dropout, natural
  sampling, unweighted BCE, AdamW at 3e-4 and weight decay 5e-4, per-step
  cosine decay to 1e-8, batch 4/GPU, fp16 two-GPU DDP with SyncBatchNorm,
  seed 88, and 10 epochs. Recreate the original non-stratified GroupKFold
  assignment for all 2,019 studies. Restore the one topology-failed study
  that the native crop manifest had omitted, retaining degenerate C6/C7
  localizations as the original pipeline did for failed coordinates. Select
  checkpoints on overall vertebra-level AUC, which is the metric the original
  code named `auc_mean`. Known source mismatches are WW1800 PNGs rather than
  WW2500 and current OOF student segmentations rather than the original
  five-fold segmentation ensemble. A two-GPU batch-4/GPU fp16 backward smoke
  test passed before launch.
- 2026-08-01: Queue a second classification approach after the X3D-L
  reproduction: the best 384-pixel MaxViT-Tiny MIL architecture with 32 MIL
  positions, where each position receives three consecutive CT slices. Match
  the close-square depth-32 experiment in crop geometry, 5% padding, and
  interpolation policy, but resample the tight crop to 96 positions before
  reshaping non-overlapping groups `(0,1,2), (3,4,5), ...` into three channels
  by 32 positions. Thus long crops are linearly reduced and short crops use
  nearest-neighbor repetition. Retain the level-specific center-slice soft
  mask as a fourth channel and keep every model, augmentation, sampling, loss,
  and optimization setting identical. Track it as a run in the shared
  `cspine_vertebra_fracture_classification` MLflow experiment.
- 2026-08-01: Insert a close-square depth-32 MaxViT-Tiny MIL experiment
  between the X3D-L reproduction and the queued stack-3 experiment. Threshold
  the level mask at 0.4 with fallback to 0.1, use tight depth bounds, and
  create a centered in-plane square around the vertebra with 5% padding per
  side. Shift boundary-touching squares rather than distort them, then linearly
  resize square-to-square at 384 pixels. Downsample depth to 32 with trilinear
  interpolation, but upsample short crops with nearest-neighbor repetition.
  Retain the established previous/current/next CT channels plus soft mask and
  all other top MaxViT-Tiny MIL settings.
- 2026-08-01: Stop the original-GroupKFold X3D-L reproduction after five
  completed epochs once epoch 4 reached overall AUC 0.9162 and macro per-level
  AUC 0.9042. This establishes that the original-style pipeline is highly
  competitive, but its fold-0 validation membership overlaps only 85 studies
  with the current project fold. Preserve its best checkpoint and relaunch the
  identical 10-epoch X3D-L preprocessing/model/optimization contract using
  `vertebra_samples_native_depth.parquet` and the current `fold` assignment.
  Keep the close-square depth-32 and matched stack-3 MaxViT experiments queued
  behind this current-fold reproduction.

## Completed experiments

For each interpretable run, record:

- objective or hypothesis;
- exact config and run ID;
- data version and split contract;
- primary and important secondary metrics;
- comparison with the relevant baseline;
- artifacts or checkpoint location; and
- disposition and link to a detailed decision-log entry when appropriate.

### 2026-07-28: Manual-only C1-C7 segmentation baseline

- Config/run: `cspine_segmentation_x3d_s/cspine-seg-x3ds-v1`
- Split: five study-level folds over all 87 manually segmented studies.
- Model: pretrained X3D-S encoder, DeepLabV3+ 3D decoder, fixed `128^3`
  input, Dice plus cross-entropy, BF16 two-GPU DDP, EMA, 100 epochs.
- Best validation mean Dice by fold: 0.6660, 0.6742, 0.6474, 0.6983,
  and 0.6867 (mean 0.6745, population SD 0.0174).
- Artifacts:
  `experiments/cspine_segmentation_x3d_s/cspine-seg-x3ds-v1/fold*/checkpoints/best.ckpt`.
- Disposition: retain as the clean manual-only comparator and OOF localizer.

### 2026-07-28: TotalSegmentator teacher benchmark

- Teacher: TotalSegmentator 2.16.0, published open `total` task, full 1.5 mm
  inference, interpolation order 3, no fast/ROI crop/small-component removal.
- Evaluation: all 87 manual cases; temporary CT and output NIfTIs deleted
  after each study.
- Mean study Dice: 0.8940 +/- 0.0170; range 0.8085-0.9165.
- Mean per-level Dice: C1 0.8831, C2 0.9142, C3 0.8919, C4 0.8884,
  C5 0.8879, C6 0.8883, C7 0.9039.
- Fracture-positive versus negative mean: 0.8961 versus 0.8922.
- Missing-level cases: 0/87.
- Disposition: use as a pseudo-label teacher, then measure pseudo-pretraining
  plus manual fine-tuning against the manual-only baseline.

## Failed or interrupted runs

Distinguish infrastructure and data-contract failures from models that trained
successfully but underperformed. A failed launch is not architecture evidence.

### 2026-08-09: Seven-token series classifier implementation and extraction

- Objective: predict `overall,C1,...,C7` jointly from one frozen MIL feature
  token per vertebra and compare LayerNorm projection versus no projection.
- Upstream families: original five-fold EMA d32 and 3x32 MaxViT-Tiny MIL
  checkpoints; fold 4 of 3x32 uses the corrected loader-fix run.
- Feature contract: 512 pooled post-BiGRU values plus bag logit, MIL logit,
  normalized attention entropy, and maximum attention; shape `(N,7,516)` with
  an explicit validity mask and learned missing token.
- Split contract: for second-stage outer fold `f`, upstream fold-`f` weights
  extract every series used in that second-stage fold. This keeps validation
  unseen by the extractor without mixing upstream embedding spaces.
- Optimization contract: natural uniformly shuffled study distribution, exact
  competition-weighted log loss, AdamW, warmup plus cosine schedule, and
  checkpoint selection by globally aggregated competition loss.
- Configs: `cspine_series_d32`, `cspine_series_d32_no_projection`,
  `cspine_series_3x32`, and `cspine_series_3x32_no_projection`.
- Status: d32 and 3x32 fold-specific cache extraction launched concurrently on
  separate RTX A6000 GPUs. Training has not started and no result is available.
- Storage: all ten caches are under the processed-data root on `/mnt/champaca`
  and total about 145 MiB before compression. Series logs and checkpoints are
  configured under `/mnt/champaca/experiments/cspine-fx`; no training artifact
  is intentionally written to the nearly full root filesystem.
- 2026-08-09 update: both fold-0 caches completed with shape `(2019,7,516)`
  and seven explicitly masked missing levels. Fold-1 extraction is active for
  both families. Four full five-fold jobs are running: `series-d32-ln-v1`,
  `series-d32-raw-v1`, `series-3x32-ln-v1`, and `series-3x32-raw-v1`. Each job
  trained fold 0 immediately; each later fold starts automatically as soon as
  its matching upstream feature cache is written. D32 fold 1 is already
  training while 3x32 fold-1 extraction continues. Fold-0 best competition loss / mean
  AUC was 0.2660/0.9443 for d32 LayerNorm, 0.2540/0.9430 for d32 raw,
  0.3084/0.9247 for 3x32 LayerNorm, and 0.2688/0.9337 for 3x32 raw. These
  lightweight series models train on CPU with four BLAS threads each so they
  do not contend with the two active GPU feature extractors. Real-cache
  forward/backward smoke tests passed for all four configurations, and the
  shared HTTP MLflow preflight passed.

### 2026-08-09: 7x32 depth-token series-classifier queue

- Added the second planned sequence granularity: 32 ordered tokens from each
  of C1-C7, for a total sequence length of 224. Each frozen MIL token contains
  the 512-dimensional contextual post-BiGRU feature, its slice logit, and its
  MIL attention weight. This deliberately excludes patient/scan metadata.
- A second BiGRU processes the complete C1-through-C7 sequence. Learned
  within-level attention pools each 32-token block for the seven level outputs;
  independent attention over all 224 positions produces the overall-fracture
  output. Learned level and within-crop depth embeddings preserve both axes.
  Entirely missing crop levels use the same explicit learned-missing-feature
  principle as the seven-token model.
- Queued five folds for d32 and 3x32, each with and without the LayerNorm-linear
  projection: `series-7x32-d32-ln-v1`, `series-7x32-d32-raw-v1`,
  `series-7x32-3x32-ln-v1`, and `series-7x32-3x32-raw-v1`. The two depth-feature
  extractors wait for their currently active pooled-feature extractor on the
  same GPU, then write only to `/mnt/champaca/.../series_features_7x32`.
  Each lightweight CPU training job starts a fold as soon as that fold cache
  appears. The competition loss, natural study distribution, folds, optimizer,
  and training schedule remain matched to the seven-token comparison.
- Focused tests pass (9/9), lint passes, both real EMA MIL checkpoints produced
  finite `(1,32,514)` features, and the shared MLflow health check passed before
  queuing.
- 2026-08-09 live update: the original seven-token heads have completed folds
  0-2. Mean best competition loss / checkpoint mean AUC over those three folds
  is 0.2860/0.9371 for d32 LayerNorm, 0.2687/0.9366 for d32 raw,
  0.3424/0.9101 for 3x32 LayerNorm, and 0.2959/0.9227 for 3x32 raw. Thus d32
  remains stronger, while no projection currently improves the primary loss
  for both upstream families. Pooled-feature extraction is on fold 3 (d32
  8,804/14,126 rows; 3x32 2,004/14,126 rows), with fold 4 remaining. The 7x32
  extractors and all four five-fold training queues are live and will begin
  automatically after those pooled jobs release their GPUs.
- On the same completed folds 0-2, level-only macro AUC clarifies the effect of
  the second stage. Original d32 MIL checkpoints average 0.9337; the seven-token
  d32 head reaches 0.9353 raw (+0.0016) and 0.9389 with LayerNorm (+0.0052).
  Original 3x32 averages 0.9286, while its raw and LayerNorm heads fall to
  0.9222 (-0.0064) and 0.9030 (-0.0256). Thus the d32 contextual sequence adds
  modest level-classification value, whereas the current 3x32 sequence heads
  do not. Checkpoint selection also differs: upstream MIL was selected on
  macro AUC, while series heads are selected on competition loss.
- 2026-08-09 five-fold OOF update: the raw d32 head reached competition loss
  0.27302, overall AUC 0.93665, and level-macro AUC 0.93307. The raw 3x32 head
  was weaker alone (0.29299/0.92269/0.92434) but complementary. Their fixed
  equal-probability ensemble reached **0.26594/0.93914/0.93833**, improving all
  seven per-level AUCs as well as overall AUC. Averaging all four raw and
  LayerNorm heads was marginally worse at 0.26628/0.93663/0.93709. OOF
  predictions and metrics are stored under
  `/mnt/champaca/experiments/cspine-fx/series_ensemble/series-7x1-oof-v1`.
- The first 7x32 training queues observed the final `.npz` path while NumPy was
  still writing its ZIP directory and exited with `BadZipFile`; all ten
  completed caches subsequently passed archive validation. Cache publication
  now writes a temporary file, flushes/fsyncs it, and atomically renames it.
  All four five-fold 7x32 jobs were relaunched successfully. Fold-0 d32
  LayerNorm reached competition loss 0.2892 and level-macro AUC 0.9374 at epoch
  2; this is an early trajectory, not a selected result.
- 2026-08-09 completed 7x32 OOF update: all four variants completed five folds.
  The raw d32 head alone reached competition loss / overall AUC / level-macro
  AUC of 0.27774/0.93037/0.92606, weaker than its 7x1 counterpart. The raw
  d32+3x32 7x32 ensemble was substantially stronger at
  **0.26124/0.93674/0.93864**, beating the selected 7x1 ensemble's 0.26594 loss
  while trading a small amount of overall AUC. Equal averaging of the 7x1 and
  7x32 raw-family ensembles produced the best result:
  **0.25981 competition loss, 0.93930 overall AUC, 0.94087 level-macro AUC**.
  The all-four 7x32 projection ensemble was inferior at 0.26303/0.93476/0.93776.
  OOF predictions and metrics are stored under
  `/mnt/champaca/experiments/cspine-fx/series_ensemble/series-7x32-oof-v1`.
- Exhaustive equal-probability evaluation of all six pairs among the four raw
  architectures selected **7x1-3x32 + 7x32-d32**: competition loss 0.26057,
  overall AUC 0.93897, and level-macro AUC 0.93851. This is only 0.00076 loss
  behind the four-model ensemble (0.25981), with half the second-stage models.
  If optimizing specifically for level discrimination instead, 7x1-d32 +
  7x32-3x32 reaches the best pairwise level-macro AUC, 0.94068, but has worse
  competition loss (0.26311) and overall AUC (0.93662). Pairwise predictions
  and metrics are stored under
  `/mnt/champaca/experiments/cspine-fx/series_ensemble/series-pairwise-oof-v1`.

### 2026-08-09: Oversized MaxViT extraction chunk interrupted

- A batch-8/chunk-256 inference optimization was attempted for both feature
  families and failed immediately because an intermediate MaxViT tensor
  exceeded PyTorch's 32-bit indexing limit.
- This was not an OOM and provides no modeling evidence. The production
  extraction uses the validated batch-4/chunk-128 setting instead.

### 2026-08-13: Hugging Face-ready end-to-end pipeline staged

- Added canonical DICOM-to-result inference with exact center-400/width-1800
  windowing, five-fold C1-C7 segmentation, exact d32/3x32 crop construction,
  all four raw series heads, and conditional d32/3x32 localization ensembling.
- Exported two versioned model directories under
  `/mnt/champaca/huggingface/cspine`: a roughly 58 MB segmentation package and
  a roughly 2.6 GB fracture-classification/localization package. All 45 model
  files use SafeTensors; optimizer state is excluded. Manifests include SHA-256
  checksums, preprocessing, ensemble definitions, evaluation metrics, and
  sanitized checkpoint provenance.
- On a real study, DICOM ingestion exactly matched the cached uint8 PNG volume
  and ordering. Both release-time classifier tensors exactly matched training
  dataset preprocessing (`max_abs_delta=0` for d32 and 3x32). Representative
  segmentation, MIL, 7x1, 7x32, and localization models loaded strictly from
  their exported weights.
- Added compact JSON/NPZ outputs and lazy restoration of localization heatmaps
  to native DICOM slice coordinates. Default inference loads folds sequentially
  and uses crop/localization batch size one to limit VRAM.
- Focused inference/pipeline/series tests pass. A full all-fold GPU timing smoke
  remains pending because both shared GPUs were occupied by an unrelated vLLM
  job; no external process was interrupted.

### 2026-11-08: Public-release reproducibility audit and first repairs

- A deep source, artifact, documentation, and clean-install review found that
  the staged inference implementation was functional but the repository did
  not yet reproduce the complete selected pipeline from public data.
- The authoritative release selection is the original d32 plus original 3x32
  MIL families, both corresponding localization decoders, and all four raw
  7x1/7x32 series heads. The slice-initialized d32 run remains a useful
  ablation but is not part of final-v1 because the downstream selected features
  and localizers were built from the original d32 checkpoints.
- Added a candidate `release/final-v1.json` contract, made segmentation export
  use the EMA state evaluated during validation, made export fail rather than
  silently reuse stale SafeTensors, and added missing clean-install
  dependencies and artifact checksum validation.
- Added deterministic teacher-QC generation. It exactly reproduced the
  existing 1,931-case pseudo-label manifest from 1,932 raw teacher cases.
- Added a CLI for constructing fracture-localization manifests. Focused tests
  pass (43/43), targeted Ruff checks pass, and `git diff --check` passes. No
  training or external upload was launched.
- Added a manifest-driven OOF evaluator and reran all 20 selected series-head
  checkpoints over 2,019 studies. Every final-ensemble AUC exactly matched the
  historical analysis; weighted log loss differed by `2.9e-10`, maximum
  aligned probability difference was `1.19e-7`, and the evaluator now caches
  each member atomically for interruption-safe resume.
- Replaced unversioned X3D Torch Hub source execution with the lockfile-pinned
  `pytorchvideo==0.1.5` implementation after confirming identical pretrained
  state hashes. Recorded X3D, four-channel MaxViT, and TotalSegmentator
  initialization hashes in the release manifest.
- Rebuilt both HF packages into a separate final-v1 candidate directory. All
  45 SafeTensors passed checksum verification and strict model loading: five
  EMA segmentation, ten EMA MIL, ten EMA localization, and twenty raw series
  states. A clean temporary environment installed only the generated wheel and
  declared inference requirements and imported the complete inference stack.
- Added a raw-versus-EMA segmentation evaluation CLI and ran it on all five
  manual-label validation folds. EMA scored 0.80367 mean Dice versus 0.80315
  for raw weights, winning four of five folds. This also corrected an older
  0.8078 summary that was inconsistent with both the selected checkpoint logs
  (0.80328 mean) and the matched re-evaluation. EMA remains the release state.
- The final publication review found that OOF mask generation loaded the raw
  state implicitly while final-v1 release inference selects EMA. The inference
  CLI now exposes an explicit `--checkpoint-state`; the reproduction command
  selects `raw` to match the crop cache used by the selected downstream models,
  while the released segmentation package remains EMA. A discriminating
  raw-versus-EMA test protects the two-state contract.
  Inference model construction also clears training-only initialization paths,
  so loading a selected checkpoint no longer depends on retaining its earlier
  pseudo-label-pretraining checkpoint at the original filesystem location.
- The review also found that series configs overrode the documented
  `MLFLOW_TRACKING_URI` with an undocumented localhost default. The override
  was removed; all four inherited series configs now use the standard
  fail-closed tracking environment contract.
- Promoted the selected pretrained-state hashes from provenance to executable
  checks. X3D-S and four-channel MaxViT verify canonical tensor-state hashes at
  construction, and the TotalSegmentator teacher verifies all three recorded
  Dataset292 artifacts after download.
