from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skp.cspine.pipeline import (
    CervicalSpineFracturePredictor,
    CervicalSpinePipeline,
    CervicalSpineSegmenter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end cervical spine fracture inference."
    )
    parser.add_argument("dicom_directory", type=Path)
    parser.add_argument("--segmenter", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-localization", action="store_true")
    parser.add_argument("--localization-threshold", type=float, default=0.5)
    parser.add_argument(
        "--localize-levels",
        help="Optional comma-separated subset such as C2,C6; overrides threshold.",
    )
    args = parser.parse_args()

    segmenter = CervicalSpineSegmenter.from_pretrained(
        args.segmenter, device=args.device
    )
    predictor = CervicalSpineFracturePredictor.from_pretrained(
        args.predictor, device=args.device
    )
    pipeline = CervicalSpinePipeline(segmenter, predictor)
    levels = (
        [value.strip().upper() for value in args.localize_levels.split(",")]
        if args.localize_levels
        else None
    )
    series, segmentation, prediction = pipeline(
        args.dicom_directory,
        localize=not args.no_localization,
        localization_threshold=args.localization_threshold,
        localize_levels=levels,
    )
    prediction.save(args.output)
    np.save(args.output / "vertebra_mask.npy", segmentation.native_mask)
    summary = {
        "StudyInstanceUID": series.study_uid,
        "SeriesInstanceUID": series.series_uid,
        "volume_shape": list(series.volume.shape),
        "segmentation_anomaly_levels": list(segmentation.anomaly_levels),
        **prediction.to_dict(),
    }
    (args.output / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["probabilities"], indent=2))


if __name__ == "__main__":
    main()
