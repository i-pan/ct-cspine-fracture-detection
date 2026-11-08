import os

from skp.configs import Config
from skp.configs.defaults import (
    dataloader_defaults,
    runtime_defaults,
    segmentation_3d_defaults,
)


cfg = Config()
runtime_defaults(cfg)
dataloader_defaults(cfg)
segmentation_3d_defaults(cfg)

work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")

cfg.project = "cspine_segmentation_x3d_s"
cfg.task = "segmentation_3d"
cfg.model = "segmentation.base_3d"
cfg.dataset = "cspine_segmentation_3d"
cfg.backbone = "x3d_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.num_classes = 8
cfg.include_last_block = False
cfg.decoder_type = "DeepLabV3PlusDecoder3d"
cfg.decoder_out_channels = 256
cfg.dim0_strides = [2, 2, 2, 1, 1]
cfg.output_stride = 8
cfg.atrous_rates = (3, 6, 10, 14)
cfg.aspp_separable = True
cfg.aspp_dropout = 0.1
cfg.seg_dropout = 0.1
cfg.activation_fn = "softmax"
cfg.normalization = "0_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0
cfg.split_column = "fold"
cfg.annotations_file = os.path.join(
    work_root, "manifests", "segmentation_samples.parquet"
)
cfg.inputs = "volume_path"
cfg.targets = "mask"
cfg.data_dir = work_root
cfg.num_slices = 128
cfg.image_height = 128
cfg.image_width = 128
cfg.output_size = (128, 128, 128)

cfg.loss = "cspine.DiceCrossEntropyLoss"
cfg.loss_params = {
    "dice_weight": 1.0,
    "ce_weight": 1.0,
    "include_background": False,
}
cfg.metrics = ["segmentation.MulticlassDiceScore"]
cfg.metric_activation_fn = "softmax"
cfg.metric_ignore_class0 = True
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.batch_size = 1
cfg.val_batch_size = 1
cfg.accumulate_grad_batches = 4
cfg.num_workers = 4
cfg.num_epochs = 100
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4, "weight_decay": 1e-2}
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
cfg.scheduler_interval = "step"
cfg.ema = {
    "on": True,
    "decay": 0.999,
    "update_after_step": 100,
    "update_every_n_steps": 1,
    "use_warmup": True,
    "warmup_gamma": 1.0,
    "warmup_power": 2 / 3,
    "switch_ema": False,
}
cfg.train_transforms = None
cfg.val_transforms = None
