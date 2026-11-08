from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VertebraMILLoss(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.bag_weight = float(params.get("bag_weight", 1.0))
        self.mil_weight = float(params.get("mil_weight", 0.25))
        pos_weight = params.get("pos_weight")
        value = [] if pos_weight is None else [float(pos_weight)]
        self.register_buffer(
            "pos_weight",
            torch.tensor(value, dtype=torch.float32),
            persistent=False,
        )

    def _bce(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.pos_weight if self.pos_weight.numel() else None
        return F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), pos_weight=weight
        )

    def forward(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        target = batch["y"]
        bag_loss = self._bce(out["logits"], target)
        mil_loss = self._bce(out["mil_logits"], target)
        return {
            "bag_loss": bag_loss,
            "mil_loss": mil_loss,
            "loss": self.bag_weight * bag_loss + self.mil_weight * mil_loss,
        }


class SeriesCompetitionLoss(nn.Module):
    """Exact RSNA competition weighting for overall,C1,...,C7 logits."""

    def __init__(self, params: dict):
        super().__init__()
        self.register_buffer(
            "negative_weights",
            torch.tensor([7.0] + [1.0] * 7, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        logits = out["logits"].float()
        target = batch["y"].float()
        if logits.shape != target.shape or logits.ndim != 2:
            raise ValueError(
                "Series competition loss expects equal (B,8) logits and targets; "
                f"got {tuple(logits.shape)} and {tuple(target.shape)}."
            )
        if logits.shape[1] != 8:
            raise ValueError(f"Expected overall,C1,...,C7 logits, got {logits.shape[1]}.")
        # Positives have twice the weight of negatives. Overall has seven
        # times the corresponding vertebral-level weight.
        weights = self.negative_weights.unsqueeze(0) * (1.0 + target)
        elementwise = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        loss = (elementwise * weights).sum() / weights.sum()
        # Keep the optimization loss name distinct from the globally aggregated
        # validation metric. Averaging batch-normalized ratios is not identical
        # to applying the competition formula after concatenating all rows.
        return {"loss": loss}


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.dice_weight = float(params.get("dice_weight", 1.0))
        self.ce_weight = float(params.get("ce_weight", 1.0))
        self.include_background = bool(params.get("include_background", False))
        class_weights = params.get("class_weights")
        weights = [] if class_weights is None else class_weights
        self.register_buffer(
            "class_weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, out: dict, batch: dict) -> dict[str, torch.Tensor]:
        logits = out["logits"].float()
        target = batch["y"].long()
        weights = self.class_weights if self.class_weights.numel() else None
        ce_loss = F.cross_entropy(logits, target, weight=weights)
        probability = logits.softmax(dim=1)
        one_hot = F.one_hot(target, num_classes=logits.size(1))
        one_hot = one_hot.movedim(-1, 1).float()
        start = 0 if self.include_background else 1
        probability = probability[:, start:]
        one_hot = one_hot[:, start:]
        spatial_dims = tuple(range(2, logits.ndim))
        intersection = (probability * one_hot).sum(dim=spatial_dims)
        denominator = (probability + one_hot).sum(dim=spatial_dims)
        dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
        valid = one_hot.sum(dim=spatial_dims) > 0
        dice_loss = 1 - dice[valid].mean()
        loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return {"dice_loss": dice_loss, "ce_loss": ce_loss, "loss": loss}
