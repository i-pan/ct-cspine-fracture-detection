"""224-token series classifier using frozen 3x32 MIL depth features."""

import copy

from skp.configs.cspine_series_7x32_d32 import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.feature_cache_template = cfg.feature_cache_template.replace("/d32/", "/3x32/")
