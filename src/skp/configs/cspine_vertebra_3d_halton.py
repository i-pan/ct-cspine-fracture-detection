"""Common fixed-volume config for the compact 3D Halton search."""

import copy
import os

from skp.configs.cspine_vertebra_x3d_s_d64_288_global import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_native_depth.parquet",
)
cfg.runtime_output_shape = (64, 288, 288)
cfg.num_slices = 64
cfg.image_height = 288
cfg.image_width = 288
cfg.num_epochs = 12
cfg.scheduler_num_epochs = 12
cfg.halton_pruning = None
