# End-to-end inference

The release is split into two Hugging Face model repositories:

1. a five-fold C1-C7 segmentation ensemble; and
2. fracture classification plus conditional localization.

Both accept either a local model directory or a Hugging Face model ID.

```python
from skp.cspine.pipeline import (
    CervicalSpineFracturePredictor,
    CervicalSpinePipeline,
    CervicalSpineSegmenter,
)

segmenter = CervicalSpineSegmenter.from_pretrained(
    "/path/to/cspine-segmentation", device="cuda"
)
predictor = CervicalSpineFracturePredictor.from_pretrained(
    "/path/to/cspine-fracture-classifier", device="cuda"
)
pipeline = CervicalSpinePipeline(segmenter, predictor)
series, segmentation, result = pipeline("/path/to/one/dicom/series")
result.save("./result")
print(result.probabilities)
```

The input directory must contain exactly one physically coherent axial CT
series. DICOM pixels are converted to Hounsfield units, clipped to `[-500,
1300]` (WL400/WW1800), rounded to uint8, and ordered using physical geometry.

`prediction.json` contains overall and C1-C7 probabilities, architecture-level
predictions, quality flags, and native slice geometry. Each selected fractured
level also receives a compressed heatmap. Use
`LevelLocalization.iter_native_heatmaps()` to restore heatmaps to native axial
slices.

This software is for research use only. It is not a medical device and must not
be used as the sole basis for diagnosis or treatment.
