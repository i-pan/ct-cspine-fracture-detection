"""Close reproduction of the 6th-place vertebra-level X3D pipeline."""

import copy
import os

from skp.configs.cspine_vertebra_x3d_s_d64_288_global import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")
cfg.seed = 88

# Original chunk contract: tight segmentation box, 64 nearest selected/repeated
# slices, axial resize ignoring aspect ratio, and grayscale-only input.
cfg.annotations_file = os.path.join(
    work_root,
    "manifests",
    "vertebra_samples_original_x3d_folds.parquet",
)
cfg.split_column = "original_fold"
cfg.original_x3d_output_shape = (64, 288, 288)
cfg.original_x3d_mask_threshold = 0.4
cfg.original_x3d_augmentation = True
cfg.runtime_output_shape = None
cfg.ct_only_3d_input = True
cfg.num_slices = 64
cfg.image_height = 288
cfg.image_width = 288

cfg.backbone = "x3d_l"
cfg.num_input_channels = 1
cfg.x3d_feature_dim = 432
cfg.global_pool_3d = True
cfg.dim0_strides = [1, 1, 1, 1, 1]
cfg.include_last_block = True
cfg.multisample_dropout = 5
cfg.enable_gradient_checkpointing = False

# The original preprocessing maps uint8 windowed CT to [-1, 1]. The existing
# PNG/crop source is WW1800; the original used WW2500 (tracked as a known
# reproduction mismatch until sufficient storage is available).
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.sampler = None
cfg.batch_size = 4
cfg.val_batch_size = 4
cfg.num_epochs = 10
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 5e-4}
cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 1e-8}
cfg.scheduler_interval = "step"
cfg.scheduler_num_epochs = 10
cfg.ema = {"on": False}

# The original checkpoint monitor called the single overall binary AUC
# `auc_mean`; retain our per-level metrics but select on the comparable AUC.
cfg.val_metric = "auc"
