from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from skp.configs.base import Config


class Net(nn.Module):
    """Contextual classifier over seven ordered 32-token vertebra sequences."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        input_dim = int(cfg.series_feature_dim)
        projection_dim = int(cfg.series_projection_dim)
        self.tokens_per_level = int(cfg.series_tokens_per_level)
        self.use_layernorm_projection = bool(cfg.use_layernorm_projection)
        if self.use_layernorm_projection:
            self.feature_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, projection_dim),
                nn.SiLU(),
            )
            model_dim = projection_dim
        else:
            self.feature_projection = nn.Identity()
            model_dim = input_dim

        self.missing_feature = nn.Parameter(torch.zeros(input_dim))
        self.level_embedding = nn.Embedding(7, model_dim)
        self.depth_embedding = nn.Embedding(self.tokens_per_level, model_dim)
        hidden_dim = int(cfg.series_hidden_dim)
        num_layers = int(cfg.series_num_layers)
        dropout = float(cfg.dropout)
        self.sequence = nn.GRU(
            input_size=model_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_dim = 2 * hidden_dim
        attention_dim = int(cfg.series_attention_dim)
        self.level_attention = nn.Sequential(
            nn.Linear(output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.overall_attention = nn.Sequential(
            nn.Linear(output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.level_head = nn.Linear(output_dim, 1)
        self.overall_head = nn.Linear(output_dim, 1)
        self.criterion = None

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_loss: bool = False,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"].float()
        mask = batch["mask"].bool()
        expected = (7, self.tokens_per_level)
        if x.ndim != 4 or x.shape[1:3] != expected:
            raise ValueError(
                f"Expected (B,7,{self.tokens_per_level},F), got {tuple(x.shape)}."
            )
        if mask.shape != x.shape[:3]:
            raise ValueError(f"Expected mask {tuple(x.shape[:3])}, got {tuple(mask.shape)}.")

        missing = self.missing_feature.view(1, 1, 1, -1)
        x = torch.where(mask.unsqueeze(-1), x, missing)
        x = self.feature_projection(x)
        level_indices = torch.arange(7, device=x.device)
        depth_indices = torch.arange(self.tokens_per_level, device=x.device)
        x = x + self.level_embedding(level_indices).view(1, 7, 1, -1)
        x = x + self.depth_embedding(depth_indices).view(
            1, 1, self.tokens_per_level, -1
        )

        batch_size = x.shape[0]
        x = x.reshape(batch_size, 7 * self.tokens_per_level, -1)
        sequence, _ = self.sequence(x)
        grouped = sequence.reshape(batch_size, 7, self.tokens_per_level, -1)

        level_attention = self.level_attention(grouped).squeeze(-1).softmax(dim=2)
        level_features = torch.sum(grouped * level_attention.unsqueeze(-1), dim=2)
        level_logits = self.level_head(self.dropout(level_features)).squeeze(-1)

        overall_attention = self.overall_attention(sequence).squeeze(-1).softmax(dim=1)
        pooled = torch.sum(sequence * overall_attention.unsqueeze(-1), dim=1)
        overall_logits = self.overall_head(self.dropout(pooled))
        logits = torch.cat([overall_logits, level_logits], dim=1)
        out = {
            "logits": logits,
            "level_attention": level_attention,
            "overall_attention": overall_attention,
        }
        if return_features:
            out.update(
                {
                    "features": pooled,
                    "features_seq": sequence,
                    "features_level": level_features,
                }
            )
        if return_loss:
            out.update(self.criterion(out, batch))
        return out

    def set_criterion(self, loss: nn.Module) -> None:
        self.criterion = loss
