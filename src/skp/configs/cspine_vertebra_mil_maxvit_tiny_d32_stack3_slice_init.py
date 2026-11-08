"""3x32 MIL initialized by the mask-free 2.5D slice student."""

import copy
from pathlib import Path

from skp.configs.cspine_vertebra_mil_maxvit_tiny_d32_stack3_loaderfix import (
    cfg as base_cfg,
)


cfg = copy.deepcopy(base_cfg)
project_root = Path(__file__).resolve().parents[3]
cfg.pretrained = False
cfg.load_pretrained_backbone = str(
    project_root
    / "experiments"
    / "cspine_slice_2dc_maxvit_tiny"
    / "cspine-slice-2dc-maxvitt-384-soft-oof-ensemble-bs32-nogc-v1"
    / "fold0"
    / "checkpoints"
    / "epoch=000-val_metric=0.9118.ckpt"
)

cfg.project = "cspine_vertebra_fracture_classification"
cfg.num_epochs = 2
cfg.scheduler_num_epochs = 2
cfg.scheduler = "LinearWarmupCosineAnnealingLR"
cfg.scheduler_params = {
    "pct_start": 0.10,
    "init_lr": 2e-6,
    "final_lr": 2e-6,
}
