from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_maxvit_tiny",
    backbone="maxvit_tiny_tf_384.in1k",
)
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
