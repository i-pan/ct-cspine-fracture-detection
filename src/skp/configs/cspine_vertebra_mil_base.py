import os

from skp.configs import Config
from skp.configs.defaults import (
    classification_2d_defaults,
    dataloader_defaults,
    runtime_defaults,
)


def build_cfg(*, project: str, backbone: str) -> Config:
    cfg = Config()
    runtime_defaults(cfg)
    dataloader_defaults(cfg)
    classification_2d_defaults(cfg)

    work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")
    cfg.project = project
    cfg.task = "classification"
    cfg.model = "classification.cspine_mil"
    cfg.dataset = "cspine_vertebra_mil"
    cfg.backbone = backbone
    cfg.pretrained = True
    cfg.num_input_channels = 4
    cfg.num_classes = 1
    cfg.image_height = 384
    cfg.image_width = 384
    cfg.num_slices = 32
    cfg.dropout = 0.2
    cfg.use_level_embedding = False
    cfg.level_embedding_dim = 16
    # No non-image inputs are exposed to the fracture classifier. The level
    # index remains in each batch solely for per-level validation metrics.
    cfg.use_metadata = False
    cfg.metadata_embedding_dim = 16
    cfg.rnn_hidden_dim = 256
    cfg.rnn_num_layers = 1
    cfg.rnn_dropout = 0.0
    cfg.attention_dim = 128
    cfg.encoder_chunk_size = 64
    cfg.mil_pool_temperature = 1.0

    cfg.fold = 0
    cfg.split_column = "fold"
    cfg.annotations_file = os.path.join(
        work_root, "manifests", "vertebra_samples.parquet"
    )
    cfg.inputs = "crop_path"
    cfg.targets = ["target"]
    cfg.data_dir = work_root
    cfg.train_transforms = None
    cfg.val_transforms = None

    cfg.loss = "cspine.VertebraMILLoss"
    cfg.loss_params = {"bag_weight": 1.0, "mil_weight": 0.25}
    cfg.metrics = ["cspine.VertebraAUROC"]
    cfg.metric_activation_fn = "sigmoid"
    cfg.val_metric = "auc_mean"
    cfg.val_track = "max"

    cfg.sampler = "WeightedSampler"
    cfg.batch_size = 2
    cfg.val_batch_size = 2
    cfg.accumulate_grad_batches = 1
    cfg.num_workers = 8
    cfg.num_epochs = 12
    cfg.optimizer = "AdamW"
    cfg.optimizer_params = {"lr": 2e-4, "weight_decay": 1e-2}
    cfg.scheduler = "LinearWarmupCosineAnnealingLR"
    cfg.scheduler_params = {
        "pct_start": 0.05,
        "init_lr": 0.0,
        "final_lr": 1e-6,
    }
    cfg.scheduler_interval = "step"
    cfg.enable_gradient_checkpointing = True
    cfg.ema = {
        "on": True,
        "decay": 0.9995,
        "update_after_step": 100,
        "update_every_n_steps": 1,
        "use_warmup": True,
        "warmup_gamma": 1.0,
        "warmup_power": 2 / 3,
        "switch_ema": False,
    }
    return cfg
