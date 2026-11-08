"""Mask-free 3-adjacent-slice classifier distilled from OOF MIL teachers."""

import os

import albumentations as A
import cv2

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

work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")
cfg.project = "cspine_slice_fracture_classification"
cfg.task = "classification"
cfg.model = "classification.net2d"
cfg.dataset = "cspine_slice_2dc"
cfg.backbone = "maxvit_tiny_tf_384.in1k"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.num_classes = 1
cfg.pool = "avg"
cfg.dropout = 0.2
cfg.backbone_img_size = False
cfg.normalization = "linear"
cfg.normalization_params = {
    "input_min": 0,
    "input_max": 255,
    "output_min": 0,
    "output_max": 1,
}

cfg.fold = 0
cfg.split_column = "fold"
cfg.annotations_file = os.path.join(
    work_root, "manifests", "slice_teacher_oof.parquet"
)
cfg.inputs = "png_path_center"
cfg.targets = ["teacher_probability"]
cfg.data_dir = work_root

cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {}
cfg.metrics = ["cspine.SliceAUROC"]
cfg.metric_activation_fn = "sigmoid"
cfg.val_metric = "auc_fold_mean"
cfg.val_track = "max"

cfg.sampler = "WeightedSampler"
cfg.pseudo_positive_threshold = 0.5
cfg.sample_positive_fraction = 0.5
cfg.batch_size = 32
cfg.val_batch_size = 64
cfg.accumulate_grad_batches = 1
cfg.num_workers = 8
cfg.val_num_workers = 4
cfg.num_epochs = 8
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 2e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"
cfg.enable_gradient_checkpointing = False
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

cfg.image_height = 384
cfg.image_width = 384
cfg.train_transforms = A.Compose(
    [
        A.Resize(cfg.image_height, cfg.image_width),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.04, 0.04),
            scale=(0.9, 1.1),
            rotate=(-8, 8),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.08, contrast_limit=0.08, p=0.3
        ),
    ]
)
cfg.val_transforms = A.Compose([A.Resize(cfg.image_height, cfg.image_width)])
