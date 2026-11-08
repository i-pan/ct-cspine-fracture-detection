from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_maxvit_tiny",
    backbone="maxvit_tiny_tf_384.in1k",
)
cfg.pretrained_state_sha256 = "4412868f5969e3da644d6d567a7640b8f860ebf4568eafaad688e54bd211b6a8"
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
