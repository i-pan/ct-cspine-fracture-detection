from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skp.cspine.crops import map_bounds_to_shape, vertebra_bounds
from skp.datasets.cspine_vertebra_mil import close_square_mil_geometry


def _parse_instances(value) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [int(item) for item in value]


def _resampled_source_indices(source_depth: int, target_depth: int) -> np.ndarray:
    if source_depth > target_depth:
        coordinates = (
            (np.arange(target_depth, dtype=np.float64) + 0.5)
            * source_depth
            / target_depth
            - 0.5
        )
        return np.clip(
            np.floor(coordinates + 0.5).astype(np.int64), 0, source_depth - 1
        )
    if source_depth < target_depth:
        return np.floor(
            np.arange(target_depth, dtype=np.float64) * source_depth / target_depth
        ).astype(np.int64)
    return np.arange(target_depth, dtype=np.int64)


def build_fracture_localization_manifest(
    vertebra_samples: pd.DataFrame,
    slices: pd.DataFrame,
    boxes: pd.DataFrame,
    mask_root: str | Path,
    *,
    target_depth: int = 32,
    crop_padding_fraction: float = 0.25,
    close_mask_threshold: float = 0.4,
    close_padding_fraction: float = 0.05,
) -> pd.DataFrame:
    """Project native boxes into close-square per-vertebra MIL crops.

    The target is one binary fracture map. Vertebral level is used only to select
    the already-built crop; it is never encoded as a localization class.
    """
    mask_root = Path(mask_root)
    train_slices = slices[slices["split"].eq("train")].copy()
    geometry = (
        train_slices.groupby("study_uid", sort=False)
        .agg(
            native_depth=("slice_index", "size"),
            native_height=("rows", "first"),
            native_width=("columns", "first"),
        )
        .to_dict("index")
    )
    boxes_by_study = {
        str(uid): frame.copy()
        for uid, frame in boxes.groupby("StudyInstanceUID", sort=False)
    }
    records = []
    candidates = vertebra_samples[
        vertebra_samples["target"].eq(1)
        & vertebra_samples["StudyInstanceUID"].astype(str).isin(boxes_by_study)
    ]
    for row in candidates.itertuples(index=False):
        uid = str(row.StudyInstanceUID)
        shape = geometry[uid]
        segmentation = np.load(mask_root / f"{uid}.npy")
        native_bounds = map_bounds_to_shape(
            vertebra_bounds(
                segmentation,
                int(row.level_index) + 1,
                crop_padding_fraction,
            ),
            segmentation.shape,
            (
                int(shape["native_depth"]),
                int(shape["native_height"]),
                int(shape["native_width"]),
            ),
        )
        expected_shape = tuple(bound.stop - bound.start for bound in native_bounds)
        source_shape = (
            int(row.source_depth),
            int(row.source_height),
            int(row.source_width),
        )
        if expected_shape != source_shape:
            raise ValueError(
                f"Reconstructed crop mismatch for {uid}/{row.level}: "
                f"{expected_shape} != {source_shape}."
            )

        with np.load(row.crop_path) as data:
            crop_mask = data["mask"].astype(np.float32)
        depth_start, depth_stop, y_start, x_start, side = close_square_mil_geometry(
            crop_mask,
            mask_threshold=close_mask_threshold,
            padding_fraction=close_padding_fraction,
        )
        source_instances = _parse_instances(row.source_instances)
        tight_instances = source_instances[depth_start:depth_stop]
        model_source_indices = _resampled_source_indices(
            len(tight_instances), target_depth
        )
        instance_to_source_index = {
            int(instance): index for index, instance in enumerate(tight_instances)
        }

        fitted_height = min(384, max(1, round(row.source_height * row.axial_scale)))
        fitted_width = min(384, max(1, round(row.source_width * row.axial_scale)))
        initial_height_padding = (384 - fitted_height) // 2
        initial_width_padding = (384 - fitted_width) // 2
        boxes_by_position: list[list[list[float]]] = [
            [] for _ in range(target_depth)
        ]
        for box in boxes_by_study[uid].itertuples(index=False):
            instance = int(box.slice_number)
            if instance not in instance_to_source_index:
                continue
            native_x0 = float(box.x)
            native_y0 = float(box.y)
            native_x1 = native_x0 + float(box.width)
            native_y1 = native_y0 + float(box.height)
            crop_x0 = max(native_x0, native_bounds[2].start)
            crop_y0 = max(native_y0, native_bounds[1].start)
            crop_x1 = min(native_x1, native_bounds[2].stop)
            crop_y1 = min(native_y1, native_bounds[1].stop)
            if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                continue

            scale = float(row.axial_scale)
            close_x0 = (
                (crop_x0 - native_bounds[2].start) * scale
                + initial_width_padding
                - x_start
            )
            close_y0 = (
                (crop_y0 - native_bounds[1].start) * scale
                + initial_height_padding
                - y_start
            )
            close_x1 = (
                (crop_x1 - native_bounds[2].start) * scale
                + initial_width_padding
                - x_start
            )
            close_y1 = (
                (crop_y1 - native_bounds[1].start) * scale
                + initial_height_padding
                - y_start
            )
            close_x0, close_x1 = np.clip([close_x0, close_x1], 0, side)
            close_y0, close_y1 = np.clip([close_y0, close_y1], 0, side)
            if close_x1 <= close_x0 or close_y1 <= close_y0:
                continue
            target_position = int(
                np.abs(
                    model_source_indices - instance_to_source_index[instance]
                ).argmin()
            )
            output_scale = 384.0 / side
            boxes_by_position[target_position].append(
                [
                    float(close_x0 * output_scale),
                    float(close_y0 * output_scale),
                    float(close_x1 * output_scale),
                    float(close_y1 * output_scale),
                ]
            )

        number_of_boxes = sum(map(len, boxes_by_position))
        if number_of_boxes == 0:
            continue
        records.append(
            {
                "StudyInstanceUID": uid,
                "level": row.level,
                "fold": int(row.fold),
                "crop_path": row.crop_path,
                "boxes_by_position": json.dumps(boxes_by_position),
                "number_of_boxes": number_of_boxes,
                "number_of_positive_positions": sum(bool(v) for v in boxes_by_position),
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("No fracture boxes intersected positive vertebra crops.")
    return result
