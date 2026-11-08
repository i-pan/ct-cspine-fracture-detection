from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_effnetv2_s_d32_bs4",
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
)
cfg.batch_size = 4
cfg.val_batch_size = 4
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
