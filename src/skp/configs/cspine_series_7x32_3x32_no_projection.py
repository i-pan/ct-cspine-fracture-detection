"""3x32 224-token series classifier without LayerNorm projection."""

import copy

from skp.configs.cspine_series_7x32_3x32 import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.use_layernorm_projection = False
