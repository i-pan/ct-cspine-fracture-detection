from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_mil_convnext_tiny",
    backbone="convnext_tiny",
)
