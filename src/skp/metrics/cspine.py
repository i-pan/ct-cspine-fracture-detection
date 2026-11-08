from __future__ import annotations

import torch
import torchmetrics as tm

from skp.metrics.classification import AUROC, _cat_state
from skp.metrics import utils


class VertebraAUROC(AUROC):
    name = "auc"

    def __init__(self, cfg, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step=dist_sync_on_step)
        self.add_state("level", default=[], dist_reduce_fx=None)

    def update(self, out: dict, batch: dict) -> None:
        super().update(out, batch)
        self.level.append(batch["level_index"].detach())

    def compute(self) -> dict[str, torch.Tensor]:
        prediction = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        target = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        levels = utils.distributed_concat(_cat_state(self.level), dim=0).cpu()
        metrics = {"auc": utils.auc(target, prediction)}
        level_values = []
        for level_index in range(7):
            selected = levels == level_index
            level_target = target[selected]
            if selected.sum() and level_target.min() != level_target.max():
                value = utils.auc(level_target, prediction[selected])
                metrics[f"auc_c{level_index + 1}"] = value
                level_values.append(value)
        metrics["auc_mean"] = (
            torch.stack(level_values).mean() if level_values else metrics["auc"]
        )
        return metrics


class SliceAUROC(AUROC):
    """Report pooled and mean-per-teacher-fold slice discrimination."""

    name = "auc"

    def __init__(self, cfg, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step=dist_sync_on_step)
        self.add_state("fold", default=[], dist_reduce_fx=None)

    def update(self, out: dict, batch: dict) -> None:
        super().update(out, batch)
        self.fold.append(batch["fold"].detach())

    def compute(self) -> dict[str, torch.Tensor]:
        prediction = utils.distributed_concat(_cat_state(self.p), dim=0).cpu()
        target = utils.distributed_concat(_cat_state(self.t), dim=0).cpu()
        folds = utils.distributed_concat(_cat_state(self.fold), dim=0).cpu()
        metrics = {
            "auc": utils.auc(target, prediction),
            "avp": utils.avp(target, prediction),
        }
        fold_aucs = []
        fold_avps = []
        for fold in sorted(torch.unique(folds).tolist()):
            selected = folds == fold
            fold_target = target[selected]
            if selected.sum() and fold_target.min() != fold_target.max():
                auc = utils.auc(fold_target, prediction[selected])
                avp = utils.avp(fold_target, prediction[selected])
                metrics[f"auc_fold{fold}"] = auc
                metrics[f"avp_fold{fold}"] = avp
                fold_aucs.append(auc)
                fold_avps.append(avp)
        metrics["auc_fold_mean"] = torch.stack(fold_aucs).mean()
        metrics["avp_fold_mean"] = torch.stack(fold_avps).mean()
        return metrics


class SeriesMetrics(tm.Metric):
    """Competition log loss plus overall and per-level discrimination."""

    def __init__(self, cfg, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.add_state("prediction", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, out: dict, batch: dict) -> None:
        self.prediction.append(out["logits"].detach().float())
        self.target.append(batch["y"].detach().float())

    def compute(self) -> dict[str, torch.Tensor]:
        logits = utils.distributed_concat(
            _cat_state(self.prediction), dim=0
        ).cpu()
        target = utils.distributed_concat(_cat_state(self.target), dim=0).cpu()
        if logits.shape != target.shape or logits.ndim != 2 or logits.shape[1] != 8:
            raise ValueError("Series metrics require equal (N,8) logits and targets.")
        probability = logits.sigmoid().clamp(1e-7, 1 - 1e-7)
        negative_weights = torch.tensor([7.0] + [1.0] * 7)
        weights = negative_weights.unsqueeze(0) * (1.0 + target)
        log_loss = -weights * (
            target * probability.log()
            + (1.0 - target) * (1.0 - probability).log()
        )
        metrics = {"competition_log_loss": log_loss.sum() / weights.sum()}
        names = ["overall"] + [f"c{level}" for level in range(1, 8)]
        level_aucs = []
        for index, name in enumerate(names):
            value = utils.auc(target[:, index], probability[:, index])
            metrics[f"auc_{name}"] = value
            if index:
                level_aucs.append(value)
        metrics["auc_levels_mean"] = torch.stack(level_aucs).mean()
        metrics["auc_mean"] = torch.stack(
            [metrics["auc_overall"], metrics["auc_levels_mean"]]
        ).mean()
        return metrics


class FractureLocalizationMetrics(tm.Metric):
    """Slice discrimination and spatial pointing for binary fracture maps."""

    def __init__(self, cfg, dist_sync_on_step: bool = False):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=False,
        )
        self.threshold = float(cfg.get("localization_metric_threshold", 0.3))
        self.add_state("prediction", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, out: dict, batch: dict) -> None:
        self.prediction.append(out["logits"].detach().sigmoid().float())
        self.target.append(batch["y"].detach().bool())

    def compute(self) -> dict[str, torch.Tensor]:
        prediction = utils.distributed_concat(_cat_state(self.prediction), dim=0).cpu()
        target = utils.distributed_concat(_cat_state(self.target), dim=0).cpu()
        slice_target = target.flatten(3).any(dim=3).flatten()
        slice_prediction = prediction.flatten(3).amax(dim=3).flatten()
        metrics = {
            "slice_auc": utils.auc(slice_target, slice_prediction),
            "slice_avp": utils.avp(slice_target, slice_prediction),
        }

        positive_maps = target.flatten(0, 2)
        predicted_maps = prediction.flatten(0, 2)
        positive = positive_maps.flatten(1).any(dim=1)
        if positive.any():
            positive_maps = positive_maps[positive].flatten(1)
            predicted_maps = predicted_maps[positive].flatten(1)
            peak_indices = predicted_maps.argmax(dim=1)
            metrics["pointing_accuracy"] = positive_maps[
                torch.arange(len(peak_indices)), peak_indices
            ].float().mean()
            binary = predicted_maps >= self.threshold
            intersection = (binary & positive_maps).sum(dim=1).float()
            denominator = binary.sum(dim=1) + positive_maps.sum(dim=1)
            metrics["positive_dice"] = (
                2 * intersection / denominator.clamp_min(1)
            ).mean()
        else:
            metrics["pointing_accuracy"] = torch.tensor(float("nan"))
            metrics["positive_dice"] = torch.tensor(float("nan"))
        return metrics
