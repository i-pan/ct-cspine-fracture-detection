from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from skp.cspine.artifacts import (
    load_artifact_model,
    read_artifact_manifest,
    resolve_pretrained_path,
)
from skp.cspine.constants import LEVELS
from skp.cspine.crops import (
    map_bounds_to_shape,
    resize_depth_and_fit_axial,
    resize_volume,
    vertebra_bounds,
)
from skp.cspine.dicom import dicom_to_hu, hu_to_uint8
from skp.cspine.segmentation_inference import (
    keep_largest_components,
    topology_anomaly_levels,
)
from skp.datasets.cspine_vertebra_mil import (
    close_square_mil_geometry,
    resample_close_square_mil_crop,
)


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def _resampled_source_indices(source_depth: int, target_depth: int) -> np.ndarray:
    if source_depth > target_depth:
        coordinates = (
            np.arange(target_depth, dtype=np.float64) + 0.5
        ) * source_depth / target_depth - 0.5
        return np.clip(
            np.floor(coordinates + 0.5).astype(np.int64), 0, source_depth - 1
        )
    if source_depth < target_depth:
        return np.floor(
            np.arange(target_depth, dtype=np.float64) * source_depth / target_depth
        ).astype(np.int64)
    return np.arange(target_depth, dtype=np.int64)


@dataclass(frozen=True)
class DICOMSeries:
    volume: np.ndarray
    study_uid: str
    series_uid: str
    instance_numbers: np.ndarray
    image_position_z: np.ndarray
    pixel_spacing: tuple[float, float]
    slice_spacing: float
    source_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.volume.ndim != 3 or self.volume.dtype != np.uint8:
            raise ValueError("volume must be uint8 with shape (D,H,W).")
        if len(self.instance_numbers) != len(self.volume):
            raise ValueError("instance_numbers must align with volume depth.")
        if len(self.image_position_z) != len(self.volume):
            raise ValueError("image_position_z must align with volume depth.")

    @classmethod
    def from_dicom_directory(cls, directory: str | Path) -> DICOMSeries:
        try:
            import pydicom
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DICOM inference requires the medical dependency group."
            ) from exc
        paths = sorted(Path(directory).expanduser().rglob("*.dcm"))
        if not paths:
            raise FileNotFoundError(f"No .dcm files found below {directory}")
        records = []
        for path in paths:
            dataset = pydicom.dcmread(str(path), force=False)
            photo = str(getattr(dataset, "PhotometricInterpretation", ""))
            if photo != "MONOCHROME2":
                raise ValueError(
                    f"Unsupported PhotometricInterpretation={photo!r} in {path}"
                )
            position = getattr(dataset, "ImagePositionPatient", None)
            z = float(position[2]) if position is not None else np.nan
            spacing = tuple(float(value) for value in dataset.PixelSpacing)
            records.append(
                {
                    "path": path,
                    "dataset": dataset,
                    "study_uid": str(dataset.StudyInstanceUID),
                    "series_uid": str(dataset.SeriesInstanceUID),
                    "instance": int(dataset.InstanceNumber),
                    "z": z,
                    "spacing": spacing,
                    "image": hu_to_uint8(dicom_to_hu(dataset)),
                }
            )
        studies = {record["study_uid"] for record in records}
        series = {record["series_uid"] for record in records}
        if len(studies) != 1 or len(series) != 1:
            raise ValueError(
                "One inference directory must contain exactly one study and one "
                f"series; got studies={len(studies)}, series={len(series)}."
            )
        shapes = {record["image"].shape for record in records}
        spacings = {record["spacing"] for record in records}
        if len(shapes) != 1 or len(spacings) != 1:
            raise ValueError("All slices must have consistent dimensions and spacing.")
        use_z = all(np.isfinite(record["z"]) for record in records)
        records.sort(
            key=(
                (lambda record: (record["z"], record["instance"]))
                if use_z
                else (lambda record: record["instance"])
            )
        )
        z_values = np.asarray([record["z"] for record in records], np.float32)
        if use_z and len(z_values) > 1:
            slice_spacing = float(np.median(np.abs(np.diff(z_values))))
        else:
            slice_spacing = float(
                getattr(records[0]["dataset"], "SpacingBetweenSlices", np.nan)
            )
        return cls(
            volume=np.stack([record["image"] for record in records]),
            study_uid=next(iter(studies)),
            series_uid=next(iter(series)),
            instance_numbers=np.asarray(
                [record["instance"] for record in records], np.int32
            ),
            image_position_z=z_values,
            pixel_spacing=next(iter(spacings)),
            slice_spacing=slice_spacing,
            source_paths=tuple(str(record["path"].resolve()) for record in records),
        )


@dataclass(frozen=True)
class SegmentationResult:
    compact_mask: np.ndarray
    native_mask: np.ndarray
    anomaly_levels: tuple[int, ...]
    fold_count: int
    compact_probabilities: np.ndarray | None = None


@dataclass(frozen=True)
class LevelCropGeometry:
    level: str
    native_bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    source_shape: tuple[int, int, int]
    axial_scale: float
    fitted_canvas_start: tuple[int, int]
    close_square: tuple[int, int, int, int, int]
    native_slice_indices: tuple[int, ...]
    native_instance_numbers: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LevelLocalization:
    level: str
    heatmap: np.ndarray
    slice_probabilities: np.ndarray
    geometry: LevelCropGeometry

    def iter_native_heatmaps(
        self, native_shape: tuple[int, int, int]
    ) -> Iterable[tuple[int, int, np.ndarray]]:
        """Yield localization heatmaps restored to native DICOM slice geometry.

        Multiple 96-position model outputs can refer to the same source slice
        after depth upsampling. Those positions are combined by voxelwise max.
        Each yielded array has ``native_shape[1:]`` and is aligned with the
        windowed CT volume returned by :class:`DICOMSeries`.
        """
        if self.heatmap.ndim != 3:
            raise ValueError(f"Expected a (D,H,W) heatmap, got {self.heatmap.shape}.")
        geometry = self.geometry
        if len(geometry.native_slice_indices) != len(self.heatmap):
            raise ValueError("Native slice mapping must align with heatmap depth.")
        if len(geometry.native_instance_numbers) != len(self.heatmap):
            raise ValueError("Instance-number mapping must align with heatmap depth.")
        native_bounds = tuple(slice(*pair) for pair in geometry.native_bounds)
        if any(bound.stop > limit for bound, limit in zip(native_bounds, native_shape)):
            raise ValueError("Localization bounds exceed the requested native shape.")

        _, _, close_y, close_x, close_side = geometry.close_square
        fitted_y, fitted_x = geometry.fitted_canvas_start
        source_height, source_width = geometry.source_shape[1:]
        fitted_height = min(384, max(1, round(source_height * geometry.axial_scale)))
        fitted_width = min(384, max(1, round(source_width * geometry.axial_scale)))

        grouped_positions: dict[int, list[int]] = {}
        for position, slice_index in enumerate(geometry.native_slice_indices):
            grouped_positions.setdefault(slice_index, []).append(position)
        for slice_index, positions in grouped_positions.items():
            crop_heatmap = self.heatmap[positions].max(axis=0)
            square = F.interpolate(
                torch.from_numpy(crop_heatmap)[None, None].float(),
                size=(close_side, close_side),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            canvas = torch.zeros((384, 384), dtype=torch.float32)
            canvas[
                close_y : close_y + close_side,
                close_x : close_x + close_side,
            ] = square
            fitted = canvas[
                fitted_y : fitted_y + fitted_height,
                fitted_x : fitted_x + fitted_width,
            ]
            restored = F.interpolate(
                fitted[None, None],
                size=(source_height, source_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            native = np.zeros(native_shape[1:], dtype=np.float32)
            native[native_bounds[1], native_bounds[2]] = restored
            instance_number = geometry.native_instance_numbers[positions[0]]
            yield slice_index, instance_number, native


@dataclass
class FracturePredictionResult:
    probabilities: dict[str, float]
    architecture_probabilities: dict[str, dict[str, float]]
    localizations: dict[str, LevelLocalization]
    missing_levels: tuple[str, ...]
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self, *, include_heatmaps: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "probabilities": self.probabilities,
            "architecture_probabilities": self.architecture_probabilities,
            "missing_levels": list(self.missing_levels),
            "quality_flags": self.quality_flags,
            "localizations": {},
        }
        for level, localization in self.localizations.items():
            item = {
                "slice_probabilities": localization.slice_probabilities.tolist(),
                "geometry": localization.geometry.to_dict(),
                "heatmap_shape": list(localization.heatmap.shape),
            }
            if include_heatmaps:
                item["heatmap"] = localization.heatmap.tolist()
            result["localizations"][level] = item
        return result

    def save(self, output_directory: str | Path) -> Path:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        (output / "prediction.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        for level, localization in self.localizations.items():
            np.savez_compressed(
                output / f"{level}_localization.npz",
                heatmap=localization.heatmap.astype(np.float16),
                slice_probabilities=localization.slice_probabilities.astype(np.float32),
                native_slice_indices=np.asarray(
                    localization.geometry.native_slice_indices, np.int32
                ),
                native_instance_numbers=np.asarray(
                    localization.geometry.native_instance_numbers, np.int32
                ),
            )
        return output


class CervicalSpineSegmenter:
    def __init__(self, model_root: Path, manifest: dict[str, Any], device: str):
        self.model_root = model_root
        self.manifest = manifest
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        device: str = "cuda",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> CervicalSpineSegmenter:
        root = resolve_pretrained_path(
            model_id_or_path, revision=revision, cache_dir=cache_dir
        )
        return cls(root, read_artifact_manifest(root, "cspine-segmentation"), device)

    @torch.inference_mode()
    def __call__(
        self,
        series_or_volume: DICOMSeries | np.ndarray,
        *,
        return_probabilities: bool = False,
    ) -> SegmentationResult:
        volume = (
            series_or_volume.volume
            if isinstance(series_or_volume, DICOMSeries)
            else np.asarray(series_or_volume)
        )
        if volume.ndim != 3:
            raise ValueError(f"Expected (D,H,W) volume, got {volume.shape}.")
        shape = tuple(self.manifest["preprocessing"]["input_shape"])
        compact = resize_volume(volume, shape, mode="trilinear")
        tensor = torch.from_numpy(compact[None, None]).float().to(self.device)
        probability_sum = np.zeros((8, *shape), np.float32)
        entries = self.manifest["models"]
        for entry in entries:
            model, _ = load_artifact_model(self.model_root, entry, device=self.device)
            with _autocast(self.device):
                probability = model({"x": tensor})["logits"].softmax(dim=1)
            probability_sum += probability[0].float().cpu().numpy()
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        probabilities = probability_sum / len(entries)
        compact_mask = keep_largest_components(
            probabilities.argmax(axis=0).astype(np.uint8)
        )
        native_mask = resize_volume(
            compact_mask, tuple(volume.shape), mode="nearest"
        ).astype(np.uint8)
        return SegmentationResult(
            compact_mask=compact_mask,
            native_mask=native_mask,
            anomaly_levels=tuple(topology_anomaly_levels(compact_mask)),
            fold_count=len(entries),
            compact_probabilities=probabilities if return_probabilities else None,
        )


@dataclass
class _PreparedCrop:
    level: str
    d32: torch.Tensor
    stack3: torch.Tensor
    geometry: LevelCropGeometry


def _make_adjacent_channels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    previous = np.concatenate([image[:1], image[:-1]], axis=0)
    following = np.concatenate([image[1:], image[-1:]], axis=0)
    return np.stack([previous, image, following, mask * 255.0], axis=0)


def _make_stack3_channels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(image) != 96:
        raise ValueError(f"Stack3 expects depth 96, got {len(image)}.")
    grouped = image.reshape(32, 3, *image.shape[1:]).transpose(1, 0, 2, 3)
    center = mask[np.arange(32) * 3 + 1]
    return np.concatenate([grouped, center[None] * 255.0], axis=0)


def _bounds_as_pairs(bounds: Iterable[slice]):
    return tuple((int(bound.start), int(bound.stop)) for bound in bounds)


def _prepare_level_crop(
    series: DICOMSeries,
    compact_mask: np.ndarray,
    level_index: int,
) -> _PreparedCrop:
    mask_bounds = vertebra_bounds(compact_mask, level_index, padding_fraction=0.25)
    native_bounds = map_bounds_to_shape(
        mask_bounds, compact_mask.shape, series.volume.shape
    )
    image_crop = series.volume[native_bounds]
    compact_binary = (compact_mask[mask_bounds] == level_index).astype(np.float32)
    mask_crop = resize_volume(compact_binary, image_crop.shape, mode="trilinear")
    fitted_image, axial_scale = resize_depth_and_fit_axial(
        image_crop,
        (image_crop.shape[0], 384, 384),
        mode="trilinear",
    )
    fitted_mask, _ = resize_depth_and_fit_axial(
        mask_crop,
        (image_crop.shape[0], 384, 384),
        mode="trilinear",
    )
    # Match the persisted native-depth crop contract used during training:
    # CT is rounded to uint8 and the soft level mask is quantized to float16
    # before the close-square runtime resampling step.
    fitted_image = np.rint(fitted_image).astype(np.uint8).astype(np.float32)
    fitted_mask = fitted_mask.astype(np.float16).astype(np.float32)
    close = close_square_mil_geometry(
        fitted_mask, mask_threshold=0.4, padding_fraction=0.05
    )
    d32_image, d32_mask = resample_close_square_mil_crop(
        fitted_image,
        fitted_mask,
        output_shape=(32, 384, 384),
        mask_threshold=0.4,
        padding_fraction=0.05,
    )
    stack_image, stack_mask = resample_close_square_mil_crop(
        fitted_image,
        fitted_mask,
        output_shape=(96, 384, 384),
        mask_threshold=0.4,
        padding_fraction=0.05,
    )
    depth_start, depth_stop, _, _, _ = close
    source_indices = (
        _resampled_source_indices(depth_stop - depth_start, 96)
        + depth_start
        + native_bounds[0].start
    )
    instance_numbers = series.instance_numbers[source_indices]
    fitted_height = min(384, max(1, round(image_crop.shape[1] * axial_scale)))
    fitted_width = min(384, max(1, round(image_crop.shape[2] * axial_scale)))
    geometry = LevelCropGeometry(
        level=LEVELS[level_index - 1],
        native_bounds=_bounds_as_pairs(native_bounds),
        source_shape=tuple(int(value) for value in image_crop.shape),
        axial_scale=float(axial_scale),
        fitted_canvas_start=((384 - fitted_height) // 2, (384 - fitted_width) // 2),
        close_square=tuple(int(value) for value in close),
        native_slice_indices=tuple(int(value) for value in source_indices),
        native_instance_numbers=tuple(int(value) for value in instance_numbers),
    )
    return _PreparedCrop(
        level=geometry.level,
        d32=torch.from_numpy(
            np.ascontiguousarray(_make_adjacent_channels(d32_image, d32_mask))
        ).float(),
        stack3=torch.from_numpy(
            np.ascontiguousarray(_make_stack3_channels(stack_image, stack_mask))
        ).float(),
        geometry=geometry,
    )


def _pooled_features(out: dict[str, torch.Tensor]) -> torch.Tensor:
    attention = out["attention"].float()
    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / np.log(attention.shape[1])
    auxiliary = torch.stack(
        [
            out["logits"].flatten().float(),
            out["mil_logits"].flatten().float(),
            entropy,
            attention.amax(dim=1),
        ],
        dim=1,
    )
    return torch.cat([out["features"].float(), auxiliary], dim=1)


def _depth_features(out: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            out["features_seq"].float(),
            out["slice_logits"].float(),
            out["attention"].float().unsqueeze(-1),
        ],
        dim=-1,
    )


class CervicalSpineFracturePredictor:
    def __init__(self, model_root: Path, manifest: dict[str, Any], device: str):
        self.model_root = model_root
        self.manifest = manifest
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        *,
        device: str = "cuda",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> CervicalSpineFracturePredictor:
        root = resolve_pretrained_path(
            model_id_or_path, revision=revision, cache_dir=cache_dir
        )
        return cls(
            root,
            read_artifact_manifest(root, "cspine-fracture-classification"),
            device,
        )

    def _prepare_crops(
        self, series: DICOMSeries, segmentation: SegmentationResult
    ) -> tuple[list[_PreparedCrop], tuple[str, ...]]:
        crops = []
        missing = []
        for level_index, level in enumerate(LEVELS, start=1):
            try:
                crops.append(
                    _prepare_level_crop(series, segmentation.compact_mask, level_index)
                )
            except ValueError:
                missing.append(level)
        if not crops:
            raise ValueError("Segmentation contains none of C1-C7; cannot classify.")
        return crops, tuple(missing)

    def _mil_forward(self, model, crop_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {}
        batch_size = int(self.manifest["inference"].get("crop_batch_size", 1))
        for chunk in crop_tensor.split(batch_size):
            with _autocast(self.device):
                result = model(
                    {"x": chunk.to(self.device, non_blocking=True)},
                    return_features=True,
                )
            for key, value in result.items():
                outputs.setdefault(key, []).append(value.float().cpu())
        return {key: torch.cat(values, dim=0) for key, values in outputs.items()}

    @torch.inference_mode()
    def __call__(
        self,
        series: DICOMSeries,
        segmentation: SegmentationResult,
        *,
        localize: bool = True,
        localization_threshold: float = 0.5,
        localize_levels: Iterable[str] | None = None,
    ) -> FracturePredictionResult:
        crops, missing = self._prepare_crops(series, segmentation)
        crop_by_level = {crop.level: crop for crop in crops}
        valid_mask = torch.tensor(
            [[level in crop_by_level for level in LEVELS]], dtype=torch.bool
        )
        architecture_predictions: dict[str, list[np.ndarray]] = {
            "7x1_d32": [],
            "7x1_3x32": [],
            "7x32_d32": [],
            "7x32_3x32": [],
        }
        for family_name, family in self.manifest["families"].items():
            crop_tensor = torch.stack(
                [
                    getattr(crop_by_level[level], family["crop_tensor"])
                    for level in LEVELS
                    if level in crop_by_level
                ]
            )
            for fold_entry in family["folds"]:
                mil_model, _ = load_artifact_model(
                    self.model_root, fold_entry["mil"], device=self.device
                )
                mil_model.encoder_chunk_size = int(
                    self.manifest["inference"]["encoder_chunk_size"]
                )
                out = self._mil_forward(mil_model, crop_tensor)
                pooled = torch.zeros(1, 7, 516)
                depth = torch.zeros(1, 7, 32, 514)
                pooled[0, valid_mask[0]] = _pooled_features(out)
                depth[0, valid_mask[0]] = _depth_features(out)
                del mil_model, out
                for representation, features in (("7x1", pooled), ("7x32", depth)):
                    head, _ = load_artifact_model(
                        self.model_root,
                        fold_entry[representation],
                        device=self.device,
                    )
                    batch = {
                        "x": features.to(self.device),
                        "mask": (
                            valid_mask.to(self.device)
                            if representation == "7x1"
                            else valid_mask[:, :, None]
                            .expand(-1, -1, 32)
                            .to(self.device)
                        ),
                    }
                    with _autocast(self.device):
                        probability = head(batch)["logits"].sigmoid()
                    architecture_predictions[f"{representation}_{family_name}"].append(
                        probability[0].float().cpu().numpy()
                    )
                    del head
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        architecture_means = {
            name: np.mean(values, axis=0)
            for name, values in architecture_predictions.items()
        }
        final = np.mean(list(architecture_means.values()), axis=0)
        names = ("overall", *LEVELS)
        probabilities = {
            name: float(value) for name, value in zip(names, final, strict=True)
        }
        architecture_probabilities = {
            architecture: {
                name: float(value) for name, value in zip(names, values, strict=True)
            }
            for architecture, values in architecture_means.items()
        }
        flags = []
        if segmentation.anomaly_levels:
            flags.append(
                "segmentation_topology_anomaly:"
                + ",".join(f"C{level}" for level in segmentation.anomaly_levels)
            )
        if missing:
            flags.append("missing_vertebrae:" + ",".join(missing))

        selected_levels: list[str] = []
        if localize:
            if localize_levels is not None:
                selected_levels = [
                    level for level in localize_levels if level in crop_by_level
                ]
            else:
                selected_levels = [
                    level
                    for level in LEVELS
                    if level in crop_by_level
                    and probabilities[level] >= localization_threshold
                ]
                if (
                    not selected_levels
                    and probabilities["overall"] >= localization_threshold
                ):
                    selected_levels = [
                        max(
                            (level for level in LEVELS if level in crop_by_level),
                            key=probabilities.__getitem__,
                        )
                    ]
                    flags.append("localized_highest_level_fallback")
        localizations = self._localize(crop_by_level, selected_levels)
        return FracturePredictionResult(
            probabilities=probabilities,
            architecture_probabilities=architecture_probabilities,
            localizations=localizations,
            missing_levels=missing,
            quality_flags=flags,
        )

    @torch.inference_mode()
    def _localize(
        self,
        crop_by_level: dict[str, _PreparedCrop],
        levels: list[str],
    ) -> dict[str, LevelLocalization]:
        if not levels:
            return {}
        heatmap_sums = {level: np.zeros((96, 96, 96), np.float32) for level in levels}
        counts = {level: 0 for level in levels}
        for family_name, family in self.manifest["families"].items():
            crop_tensor = torch.stack(
                [
                    getattr(crop_by_level[level], family["crop_tensor"])
                    for level in levels
                ]
            )
            for fold_entry in family["folds"]:
                localizer, _ = load_artifact_model(
                    self.model_root, fold_entry["localizer"], device=self.device
                )
                logits_parts = []
                batch_size = int(
                    self.manifest["inference"].get("localization_batch_size", 1)
                )
                for chunk in crop_tensor.split(batch_size):
                    with _autocast(self.device):
                        logits_parts.append(
                            localizer({"x": chunk.to(self.device)})["logits"]
                            .float()
                            .cpu()
                        )
                logits = torch.cat(logits_parts)
                if family_name == "d32":
                    common = F.interpolate(
                        logits.sigmoid(),
                        size=(96, 96, 96),
                        mode="trilinear",
                        align_corners=False,
                    )[:, 0]
                else:
                    common = (
                        logits.sigmoid().permute(0, 2, 1, 3, 4).reshape(-1, 96, 96, 96)
                    )
                for level, heatmap in zip(levels, common.cpu().numpy(), strict=True):
                    heatmap_sums[level] += heatmap
                    counts[level] += 1
                del localizer
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        result = {}
        for level in levels:
            heatmap = heatmap_sums[level] / counts[level]
            result[level] = LevelLocalization(
                level=level,
                heatmap=heatmap.astype(np.float32),
                slice_probabilities=heatmap.reshape(96, -1).max(axis=1),
                geometry=crop_by_level[level].geometry,
            )
        return result


class CervicalSpinePipeline:
    def __init__(
        self,
        segmenter: CervicalSpineSegmenter,
        predictor: CervicalSpineFracturePredictor,
    ):
        self.segmenter = segmenter
        self.predictor = predictor

    def __call__(
        self,
        dicom_directory_or_series: str | Path | DICOMSeries,
        **prediction_kwargs,
    ) -> tuple[DICOMSeries, SegmentationResult, FracturePredictionResult]:
        series = (
            dicom_directory_or_series
            if isinstance(dicom_directory_or_series, DICOMSeries)
            else DICOMSeries.from_dicom_directory(dicom_directory_or_series)
        )
        segmentation = self.segmenter(series)
        prediction = self.predictor(series, segmentation, **prediction_kwargs)
        return series, segmentation, prediction
