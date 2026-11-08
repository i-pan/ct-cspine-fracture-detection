# Final-v1 pipeline

`release/final-v1.json` is the machine-readable source of truth for selected
configs, run names, checkpoint states, fold exceptions, ensemble members, and
reported metrics.

## Stages

```text
RSNA DICOM
  -> WL400/WW1800 native-resolution PNGs and study folds
  -> TotalSegmentator 2.16.0 C1-C7 pseudo-labels
  -> five-fold X3D-S segmentation student
  -> compact OOF C1-C7 masks with topology fallback
  -> native-depth square C1-C7 crops
  -> d32 and 3x32 MaxViT-Tiny MIL classifiers
  -> binary d32 and 3x32 localization decoders
  -> 7x1 and 7x32 frozen-MIL feature sequences
  -> four eight-output series heads
  -> equal probability ensemble
```

## Segmentation

The segmentation student receives one `128 x 128 x 128` windowed grayscale
volume and predicts background plus C1-C7. It is pretrained on 1,931
TotalSegmentator pseudo-labels, then fine-tuned on the 87 manual masks. The
five released folds use EMA weights because validation and checkpoint selection
also use EMA.

For crop generation, each study is initially inferred by the model that held
out its fold. Topology failures are retried with the five-fold ensemble. Masks
are saved on the compact grid and mapped back to native coordinates when crops
are constructed.

## Per-vertebra classification

Both selected MIL models use a close square crop at `384 x 384` axial
resolution. Every position contains three CT channels plus the target
vertebra's soft mask channel.

- **d32:** 32 adjacent depth positions after resampling the complete crop.
- **3x32:** resample to 96 slices, then reshape non-overlapping triples into
  three channels across 32 positions.

Each MaxViT-Tiny encoder feeds a bidirectional GRU, per-position fracture
logits, attention pooling, and a crop-level binary output. The final release
uses the original ImageNet-initialized MIL checkpoints. Slice-pseudolabel
initialization is retained as an ablation only.

## Series classification

Each study supplies C1-C7 in anatomical order. Four lightweight sequence heads
predict `[overall, C1, ..., C7]`:

- d32 pooled 7x1 features;
- d32 depth-resolved 7x32 features;
- 3x32 pooled 7x1 features; and
- 3x32 depth-resolved 7x32 features.

All four heads omit the LayerNorm projection ablation. Their five-fold
probabilities are averaged within architecture, then the four architecture
probabilities are averaged equally. The loss is the competition-weighted log
loss under the natural study distribution.

## Localization

The frozen image encoder from each original MIL family feeds a binary decoder
trained only on available fracture boxes. It is invoked for a predicted
fractured vertebral level and produces a weakly supervised `96 x 96 x 96`
heatmap. Vertebral identity is not a localization class. Heatmaps are
localization aids, not pixel-accurate fracture segmentations.
