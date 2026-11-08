"""d32 series classifier without the LayerNorm projection block."""

import copy

from skp.configs.cspine_series_d32 import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.use_layernorm_projection = False
