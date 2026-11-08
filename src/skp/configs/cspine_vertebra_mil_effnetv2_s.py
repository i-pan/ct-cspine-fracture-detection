from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_effnetv2_s",
    backbone="tf_efficientnetv2_s.in21k_ft_in1k",
)
