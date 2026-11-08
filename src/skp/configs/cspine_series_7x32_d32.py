"""224-token series classifier using frozen d32 MIL depth features."""

import copy

from skp.configs.cspine_series_d32 import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.model = "classification.cspine_series_depth"
cfg.dataset = "cspine_series_depth_features"
cfg.feature_cache_template = cfg.feature_cache_template.replace(
    "/series_features/", "/series_features_7x32/"
)
cfg.series_feature_dim = 514
cfg.series_tokens_per_level = 32
