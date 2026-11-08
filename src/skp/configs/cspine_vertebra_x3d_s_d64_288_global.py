import copy
import os

from skp.configs.cspine_vertebra_mil_x3d_s import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.num_slices = 64
cfg.image_height = 288
cfg.image_width = 288
cfg.global_pool_3d = True
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_d64_288.parquet",
)
