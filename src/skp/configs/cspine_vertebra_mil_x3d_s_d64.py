import copy
import os

from skp.configs.cspine_vertebra_mil_x3d_s import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.num_slices = 64
cfg.variable_depth = True
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_d64_variable.parquet",
)
