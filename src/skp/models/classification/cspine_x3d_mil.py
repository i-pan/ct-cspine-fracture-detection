from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from skp.configs.base import Config
from skp.models.classification.mil_3d import log_sum_exp_pool
from skp.models.encoders3d import get_encoder
from skp.models.normalization import normalize_input
from skp.models.utils import filter_weights_by_prefix, torch_load_weights


class Net(nn.Module):
    """Depth-preserving 3D encoder with slice-grid LSE aggregation."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = get_encoder(cfg)
        self._verify_temporal_strides()
        feature_dim = int(cfg.get("x3d_feature_dim", 432))
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.2)))
        self.linear = nn.Linear(feature_dim, cfg.num_classes)
        self.global_pool_3d = bool(cfg.get("global_pool_3d", False))
        multisample_dropout = cfg.get("multisample_dropout", False)
        if isinstance(multisample_dropout, bool):
            self.num_dropout_samples = 5 if multisample_dropout else 1
        else:
            self.num_dropout_samples = max(1, int(multisample_dropout))
        self.temperature = float(cfg.get("mil_pool_temperature", 0.5))
        self.normalize_lse = bool(cfg.get("mil_pool_normalize", True))
        self.criterion = None

        if cfg.get("load_pretrained_model"):
            weights = torch_load_weights(cfg.load_pretrained_model)
            self.load_state_dict(filter_weights_by_prefix(weights, "model."))

    def _verify_temporal_strides(self) -> None:
        temporal_strides = [
            module.stride[0]
            for module in self.encoder.modules()
            if isinstance(module, nn.Conv3d)
        ]
        if any(stride != 1 for stride in temporal_strides):
            raise ValueError(
                "3D MIL requires every convolutional depth stride to be 1; "
                f"got {temporal_strides}."
            )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = normalize_input(batch["x"], self.cfg)
        input_depth = x.shape[2]
        feature_map = self.encoder(x)[-1]
        if feature_map.shape[2] != input_depth:
            raise RuntimeError(
                "The 3D encoder changed the depth grid from "
                f"{input_depth} to {feature_map.shape[2]}."
            )
        if self.global_pool_3d:
            features = feature_map.mean(dim=(2, 3, 4))
            logits = torch.stack(
                [
                    self.linear(self.dropout(features))
                    for _ in range(self.num_dropout_samples)
                ]
            ).mean(dim=0)
            out = {"logits": logits}
            if return_features:
                out["features"] = features
                out["feature_map"] = feature_map
            if return_loss:
                out.update(self.criterion(out, batch))
            return out
        # Preserve D while collapsing only the two axial spatial dimensions.
        slice_features = feature_map.mean(dim=(-2, -1)).transpose(1, 2)
        slice_logits = self.linear(self.dropout(slice_features))
        logits = log_sum_exp_pool(
            slice_logits,
            mask=batch.get("mask"),
            temperature=self.temperature,
            normalize=self.normalize_lse,
        )
        out = {"logits": logits, "slice_logits": slice_logits}
        if return_features:
            out["slice_features"] = slice_features
            out["feature_map"] = feature_map
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
