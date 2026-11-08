"""Seven-token series classifier using frozen d32 MIL features."""

import os

from skp.configs import Config
from skp.configs.defaults import (
    classification_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
classification_2d_defaults(cfg)

work_root = os.environ.get(
    "CSPINE_WORK_ROOT",
    "./data/processed",
)
cfg.save_dir = os.environ.get(
    "CSPINE_EXPERIMENT_ROOT", "./experiments"
)
cfg.project = "cspine_series_fracture_classification"
cfg.task = "classification"
cfg.model = "classification.cspine_series"
cfg.dataset = "cspine_series_features"
cfg.fold = 0
cfg.split_column = "fold"
cfg.feature_cache_template = os.path.join(
    work_root, "series_features", "d32", "fold{fold}.npz"
)
cfg.series_feature_dim = 516
cfg.use_layernorm_projection = True
cfg.series_projection_dim = 256
cfg.series_hidden_dim = 256
cfg.series_num_layers = 1
cfg.series_attention_dim = 128
cfg.dropout = 0.2
cfg.num_classes = 8

cfg.loss = "cspine.SeriesCompetitionLoss"
cfg.loss_params = {}
cfg.metrics = ["cspine.SeriesMetrics"]
cfg.metric_activation_fn = "sigmoid"
cfg.val_metric = "competition_log_loss"
cfg.val_track = "min"

# Preserve the natural study distribution. The competition weights are
# applied inside the loss, never through a sampler.
cfg.sampler = None
cfg.batch_size = 128
cfg.val_batch_size = 256
cfg.num_workers = 4
cfg.num_epochs = 30
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.1, "init_lr": 3e-6, "final_lr": 3e-6}
cfg.scheduler_interval = "step"
cfg.early_stopping = True
cfg.early_stopping_patience = 8
cfg.ema = {"on": False}
