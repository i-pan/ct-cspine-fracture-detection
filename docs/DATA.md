# Data contract

## Source

This project uses the public **RSNA 2022 Cervical Spine Fracture Detection**
competition dataset. Users must accept the competition rules and download the
data from Kaggle themselves:

<https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection>

The repository does not redistribute DICOMs, PNGs, NIfTI files, masks,
annotations, or generated manifests. The competition data is subject to its
own access and non-commercial-use terms; the repository's Apache-2.0 license
applies to source code, not to the dataset.

## Expected source layout

Set `CSPINE_DATA_ROOT` to the extracted competition directory:

```text
$CSPINE_DATA_ROOT/
├── train.csv
├── train_bounding_boxes.csv
├── train_images/<StudyInstanceUID>/*.dcm
├── test_images/<StudyInstanceUID>/*.dcm
└── segmentations/<StudyInstanceUID>.nii
```

The release contract expects:

- 2,019 training studies;
- 711,601 training DICOM slices;
- 87 studies with manual C1-C7 segmentation;
- 235 studies represented in the fracture bounding-box table; and
- five study-level folds generated with seed 2022.

The fold is assigned at `StudyInstanceUID` level. Slices and vertebral crops
from one study must never cross folds.

## Generated data

Set `CSPINE_WORK_ROOT` to a filesystem with enough capacity. The pipeline
creates native-resolution WL400/WW1800 PNGs, Parquet manifests, compact
segmentation samples, OOF vertebral masks, and per-vertebra NPZ crops. All are
ignored by Git.

Expected release-stage counts are recorded in
[`release/final-v1.json`](../release/final-v1.json):

- 1,932 raw TotalSegmentator pseudo-label cases;
- 1,931 cases after rejecting the one all-background teacher result;
- 14,126 native-depth vertebral crops; and
- 457 positive boxed vertebral crops for each localization representation.

Generated manifests contain absolute local paths. They are reproducible build
artifacts and must not be committed or uploaded as dataset substitutes.
