import os

from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_effnetv2_s",
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
)
cfg.num_slices = 64
cfg.variable_depth = True
# A local batch contains at most 2 * 64 valid frames. Keeping it in one
# encoder call avoids rank-dependent chunk counts under DDP.
cfg.encoder_chunk_size = 128
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_d64_variable.parquet",
)
