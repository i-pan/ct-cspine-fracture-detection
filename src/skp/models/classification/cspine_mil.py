from __future__ import annotations

from typing import Dict
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
from timm import create_model

from skp.configs.base import Config
from skp.models.classification.mil_3d import log_sum_exp_pool
from skp.models.utils import (
    filter_weights_by_prefix,
    torch_load_weights,
    verify_state_dict_sha256,
)


class Net(nn.Module):
    """Level-aware 2.5D CNN-BiGRU MIL classifier for one vertebra crop."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=cfg.num_input_channels,
        )
        expected_hash = cfg.get("pretrained_state_sha256")
        if cfg.pretrained and expected_hash:
            verify_state_dict_sha256(
                self.backbone.state_dict(), expected_hash, name=cfg.backbone
            )
        if cfg.get("enable_gradient_checkpointing", False):
            self.backbone.set_grad_checkpointing()

        with torch.no_grad():
            sample = torch.zeros(
                2, cfg.num_input_channels, cfg.image_height, cfg.image_width
            )
            self.feature_dim = int(self.backbone(sample).shape[1])

        self.use_level_embedding = bool(cfg.get("use_level_embedding", False))
        level_dim = (
            int(cfg.get("level_embedding_dim", 16))
            if self.use_level_embedding
            else 0
        )
        self.use_metadata = bool(cfg.get("use_metadata", False))
        self.level_embedding = (
            nn.Embedding(7, level_dim) if self.use_level_embedding else None
        )
        metadata_dim = (
            int(cfg.get("metadata_embedding_dim", 16)) if self.use_metadata else 0
        )
        self.metadata_encoder = (
            nn.Sequential(
                nn.Linear(4, metadata_dim),
                nn.LayerNorm(metadata_dim),
                nn.SiLU(),
            )
            if self.use_metadata
            else None
        )
        sequence_dim = self.feature_dim + level_dim + metadata_dim
        hidden_dim = int(cfg.get("rnn_hidden_dim", 256))
        self.rnn = nn.GRU(
            sequence_dim,
            hidden_dim,
            num_layers=int(cfg.get("rnn_num_layers", 1)),
            batch_first=True,
            bidirectional=True,
            dropout=(
                float(cfg.get("rnn_dropout", 0.0))
                if int(cfg.get("rnn_num_layers", 1)) > 1
                else 0.0
            ),
        )
        rnn_dim = hidden_dim * 2
        attention_dim = int(cfg.get("attention_dim", 128))
        self.attention = nn.Sequential(
            nn.Linear(rnn_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        dropout = float(cfg.get("dropout", 0.2))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_dim, 1)
        self.slice_classifier = nn.Linear(self.feature_dim, 1)
        self.mil_temperature = float(cfg.get("mil_pool_temperature", 1.0))
        self.encoder_chunk_size = int(cfg.get("encoder_chunk_size", 16))
        self.criterion = None

        if cfg.get("load_pretrained_backbone"):
            self.load_pretrained_backbone()
        if cfg.get("load_pretrained_model"):
            weights = torch_load_weights(cfg.load_pretrained_model)
            self.load_state_dict(filter_weights_by_prefix(weights, "model."))

    def load_pretrained_backbone(self) -> None:
        """Load a 2D encoder, zero-initializing any new input channels."""
        weights = torch_load_weights(self.cfg.load_pretrained_backbone)
        weights = filter_weights_by_prefix(weights, "model.backbone.")
        current = self.backbone.state_dict()
        adapted = []
        for key, value in list(weights.items()):
            target = current.get(key)
            if target is None or value.shape == target.shape:
                continue
            if value.numel() == target.numel():
                weights[key] = value.reshape(target.shape)
                adapted.append(key)
                continue
            same_convolution = (
                value.ndim == target.ndim == 4
                and value.shape[0] == target.shape[0]
                and value.shape[2:] == target.shape[2:]
                and value.shape[1] < target.shape[1]
            )
            if not same_convolution:
                raise ValueError(
                    f"Cannot adapt backbone weight {key}: "
                    f"{tuple(value.shape)} -> {tuple(target.shape)}"
                )
            expanded = value.new_zeros(target.shape)
            expanded[:, : value.shape[1]] = value
            weights[key] = expanded
            adapted.append(key)
        missing, unexpected = self.backbone.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Pretrained backbone does not match target: "
                f"missing={missing}, unexpected={unexpected}"
            )
        print(
            f"Loaded pretrained backbone from {self.cfg.load_pretrained_backbone}; "
            f"adapted backbone weights: {adapted}"
        )

    def encode_slices(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got {tuple(x.shape)}")
        x = x.float() / 255.0
        batch, channels, depth, height, width = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, channels, height, width
        )
        flat_mask = None
        if valid_mask is not None:
            if valid_mask.shape != (batch, depth):
                raise ValueError(
                    f"Expected mask {(batch, depth)}, got {tuple(valid_mask.shape)}"
                )
            flat_mask = valid_mask.bool().reshape(-1)
            flat = flat[flat_mask]
        if len(flat) == 0:
            raise ValueError("Every batch must contain at least one valid slice.")
        number_of_chunks = math.ceil(len(flat) / self.encoder_chunk_size)
        base_size, extra = divmod(len(flat), number_of_chunks)
        chunk_sizes = [
            base_size + (index < extra) for index in range(number_of_chunks)
        ]
        chunks = [
            self.backbone(chunk) for chunk in flat.split(chunk_sizes, dim=0)
        ]
        features = torch.cat(chunks, dim=0)
        if flat_mask is not None:
            all_features = features.new_zeros(
                batch * depth,
                self.feature_dim,
            )
            all_features[flat_mask] = features
            features = all_features
        return features.reshape(batch, depth, self.feature_dim)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        valid_mask = batch.get("mask")
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
        features = self.encode_slices(batch["x"], valid_mask)
        batch_size, depth, _ = features.shape
        sequence_parts = [features]
        if self.level_embedding is not None:
            level = self.level_embedding(batch["level_index"]).unsqueeze(1)
            sequence_parts.append(level.expand(batch_size, depth, -1))
        if self.metadata_encoder is not None:
            metadata = self.metadata_encoder(batch["metadata"].float()).unsqueeze(1)
            sequence_parts.append(metadata.expand(batch_size, depth, -1))
        sequence = torch.cat(sequence_parts, dim=-1)
        if valid_mask is None:
            sequence, _ = self.rnn(sequence)
        else:
            lengths = valid_mask.sum(dim=1)
            if torch.any(lengths == 0):
                raise ValueError("Every sample must contain at least one valid slice.")
            packed = pack_padded_sequence(
                sequence,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed, _ = self.rnn(packed)
            sequence, _ = pad_packed_sequence(
                packed,
                batch_first=True,
                total_length=depth,
            )

        attention_logits = self.attention(sequence).squeeze(-1)
        if valid_mask is not None:
            attention_logits = attention_logits.masked_fill(
                ~valid_mask, -torch.inf
            )
        attention = attention_logits.softmax(dim=1)
        pooled = torch.sum(sequence * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(pooled))

        slice_logits = self.slice_classifier(features)
        mil_logits = log_sum_exp_pool(
            slice_logits,
            mask=valid_mask,
            temperature=self.mil_temperature,
            normalize=True,
        )
        out = {
            "logits": logits,
            "mil_logits": mil_logits,
            "slice_logits": slice_logits,
            "attention": attention,
        }
        if return_features:
            out["features"] = pooled
            out["slice_features"] = features
            # Expose the contextual per-depth representation for frozen
            # second-stage sequence models. This does not alter MIL logits or
            # checkpoint state; it only makes the already-computed sequence
            # available during feature extraction.
            out["features_seq"] = sequence
        if return_loss:
            losses = self.criterion(out, batch)
            out.update(losses)
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
