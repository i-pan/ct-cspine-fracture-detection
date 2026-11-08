"""Top MaxViT MIL model with close square, hybrid-resampled crops."""

import copy
import os

from skp.configs.cspine_vertebra_mil_maxvit_tiny import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
cfg.annotations_file = os.path.join(
    os.path.dirname(cfg.annotations_file),
    "vertebra_samples_native_depth.parquet",
)
cfg.close_square_mil_output_shape = (32, 384, 384)
cfg.close_square_mil_mask_threshold = 0.4
cfg.close_square_mil_padding_fraction = 0.05
cfg.num_slices = 32
cfg.image_height = 384
cfg.image_width = 384
cfg.num_input_channels = 4
cfg.project = "cspine_vertebra_fracture_classification"
