import copy

from skp.configs.cspine_vertebra_mil_x3d_s import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.backbone = "x3d_l"
