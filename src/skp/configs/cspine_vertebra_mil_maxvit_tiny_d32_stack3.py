"""Close-square MaxViT MIL over 32 non-overlapping three-slice groups."""

import copy

from skp.configs.cspine_vertebra_mil_maxvit_tiny_d32_close_square import (
    cfg as base_cfg,
)


cfg = copy.deepcopy(base_cfg)

# Match the close-square depth-32 experiment in every respect except its slice
# representation: resample the tight square crop to 96 positions using the
# same linear-down/nearest-up policy, then reshape consecutive triplets into
# (channels=3, depth=32). Retain the center-position soft mask as channel four.
cfg.close_square_mil_output_shape = (96, 384, 384)
cfg.num_slices = 32
cfg.spaced_stack_size = 3
cfg.num_input_channels = 4
