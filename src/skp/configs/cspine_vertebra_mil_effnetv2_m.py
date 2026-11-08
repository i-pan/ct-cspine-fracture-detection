from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_effnetv2_m",
    backbone="tf_efficientnetv2_m.in21k_ft_in1k",
)
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
