import os

from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_maxvit_tiny_512",
    backbone="maxvit_tiny_tf_512.in1k",
)
cfg.image_height = 512
cfg.image_width = 512
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_d32_512.parquet",
)
