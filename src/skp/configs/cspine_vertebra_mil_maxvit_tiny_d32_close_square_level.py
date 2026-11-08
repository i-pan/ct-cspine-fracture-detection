"""Close-square depth-32 MaxViT MIL with vertebral-level conditioning."""

import copy

from skp.configs.cspine_vertebra_mil_maxvit_tiny_d32_close_square import (
    cfg as base_cfg,
)


cfg = copy.deepcopy(base_cfg)

# Matched against the image-only close-square run. The C1-C7 identity is the
# sole non-image input; acquisition metadata remains disabled.
cfg.use_level_embedding = True
cfg.use_metadata = False
