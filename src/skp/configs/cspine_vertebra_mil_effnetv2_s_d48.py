import os

from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_effnetv2_s_d48",
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
)
cfg.num_slices = 48
cfg.variable_depth = True
# A local 2/GPU batch contains at most 2 * 48 valid frames. One invocation
# avoids rank-dependent encoder chunk counts under DDP.
cfg.encoder_chunk_size = 96
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_d48_variable.parquet",
)
