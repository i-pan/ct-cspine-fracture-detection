import os

from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_effnetv2_s_d16_stack5",
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
)
cfg.num_slices = 16
cfg.spaced_stack_size = 5
cfg.num_input_channels = 6
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_native_depth.parquet",
)
