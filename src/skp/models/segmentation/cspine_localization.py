from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from timm import create_model

from skp.models.segmentation.decoders.unet import UnetDecoder
from skp.models.utils import filter_weights_by_prefix, torch_load_weights


class Net(nn.Module):
    """Frozen MIL image encoder with a shared binary spatial decoder."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = create_model(
            cfg.backbone,
            pretrained=False,
            features_only=True,
            in_chans=cfg.num_input_channels,
        )
        with torch.no_grad():
            features = self.encoder(
                torch.zeros(1, cfg.num_input_channels, cfg.image_height, cfg.image_width)
            )
        cfg.encoder_channels = [feature.shape[1] for feature in features]
        self.decoder = UnetDecoder(cfg)
        self.segmentation_head = nn.Conv2d(
            cfg.decoder_out_channels[-1], cfg.num_classes, kernel_size=1
        )
        self.criterion = None
        checkpoints = cfg.get("mil_checkpoint_by_fold") or {}
        checkpoint = checkpoints.get(int(cfg.fold), cfg.load_pretrained_encoder)
        if checkpoint:
            self._load_mil_encoder(checkpoint)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def _load_mil_encoder(self, checkpoint: str) -> None:
        weights = torch_load_weights(checkpoint)
        weights = filter_weights_by_prefix(weights, "model.backbone.")
        weights = {
            key.replace("stages.", "stages_"): value
            for key, value in weights.items()
            if not key.startswith("head.")
        }
        missing, unexpected = self.encoder.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise ValueError(
                "MIL checkpoint does not exactly initialize the localization "
                f"encoder: missing={missing}, unexpected={unexpected}"
            )
        print(f"Loaded and froze MIL encoder from {checkpoint}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self, batch: Dict[str, torch.Tensor], return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"].float() / 255.0
        batch_size, channels, depth, height, width = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * depth, channels, height, width
        )
        with torch.no_grad():
            feature_maps = self.encoder(flat)
        decoded = self.decoder(feature_maps)[-1]
        logits = self.segmentation_head(decoded)
        logits = logits.reshape(
            batch_size, depth, self.cfg.num_classes, *logits.shape[-2:]
        ).permute(0, 2, 1, 3, 4)
        out = {"logits": logits}
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
