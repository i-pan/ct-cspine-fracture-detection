"""Binary fracture heatmaps from the frozen original d32 MIL encoder."""

import os

from skp.configs import Config
from skp.configs.defaults import (
    dataloader_defaults,
    runtime_defaults,
    segmentation_2d_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
segmentation_2d_defaults(cfg)

work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")
cfg.project = "cspine_fracture_localization"
cfg.task = "segmentation_2d"
cfg.model = "segmentation.cspine_localization"
cfg.dataset = "cspine_fracture_localization"
cfg.annotations_file = os.path.join(
    work_root, "manifests", "fracture_localization_d32.parquet"
)
cfg.inputs = "crop_path"
cfg.targets = "boxes_by_position"
cfg.split_column = "fold"
cfg.fold = 0

cfg.backbone = "maxvit_tiny_tf_384.in1k"
cfg.pretrained = False
cfg.num_input_channels = 4
cfg.num_classes = 1
cfg.image_height = 384
cfg.image_width = 384
cfg.close_square_mil_output_shape = (32, 384, 384)
cfg.close_square_mil_mask_threshold = 0.4
cfg.close_square_mil_padding_fraction = 0.05
cfg.localization_target_size = 96

cfg.load_pretrained_encoder = os.path.abspath(
    "experiments/cspine_vertebra_mil_maxvit_tiny_d32_close_square/"
    "cspine-vertebra-maxvitt-mil-d32-close-square-imageonly-bs2-v1/"
    "fold0/checkpoints/best.ckpt"
)
cfg.freeze_encoder = True
cfg.decoder_type = "UnetDecoder"
cfg.decoder_out_channels = [256, 128, 64]
cfg.decoder_n_blocks = 3
# The decoder sees B*D=32 images even with one crop/GPU, so BatchNorm has a
# stable effective batch and is compatible with the template's SyncBN path.
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.decoder_separable_conv = False

cfg.loss = "segmentation.DiceFocalLoss"
cfg.loss_params = {
    "compute_method": "per_batch",
    "activation_fn": "sigmoid",
    "dice_weight": 1.0,
    "focal_weight": 1.0,
    "gamma": 2.0,
    "alpha": 0.75,
    "scale": 10.0,
}
cfg.metrics = ["cspine.FractureLocalizationMetrics"]
cfg.activation_fn = "sigmoid"
cfg.metric_activation_fn = "sigmoid"
cfg.localization_metric_threshold = 0.3
cfg.val_metric = "slice_avp"
cfg.val_track = "max"

cfg.batch_size = 4
cfg.val_batch_size = 4
cfg.num_workers = 4
cfg.accumulate_grad_batches = 1
cfg.num_epochs = 12
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {
    "pct_start": 0.1,
    "init_lr": 3e-6,
    "final_lr": 3e-6,
}
cfg.scheduler_interval = "step"
cfg.ema = {
    "on": True,
    "decay": 0.99,
    "update_after_step": 20,
    "update_every_n_steps": 1,
    "use_warmup": True,
    "warmup_gamma": 1.0,
    "warmup_power": 2 / 3,
    "switch_ema": False,
}
