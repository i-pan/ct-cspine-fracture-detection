"""3x32 MaxViT MIL with a bounded, non-persistent CPU input pipeline."""

import copy

from skp.configs.cspine_vertebra_mil_maxvit_tiny_d32_stack3 import (
    cfg as base_cfg,
)


cfg = copy.deepcopy(base_cfg)

# Runtime 96-slice interpolation creates large temporary tensors. A small
# worker pool avoids the thread and allocator churn observed late in the
# original sequential cross-validation process.
cfg.num_workers = 2
cfg.val_num_workers = 2
cfg.persistent_workers = False
