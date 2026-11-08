"""Three binary slice heatmaps per position from the frozen 3x32 MIL encoder."""

import copy
import os

from skp.configs.cspine_fracture_localization_d32 import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")
cfg.annotations_file = os.path.join(
    work_root, "manifests", "fracture_localization_3x32.parquet"
)
cfg.close_square_mil_output_shape = (96, 384, 384)
cfg.spaced_stack_size = 3
cfg.num_classes = 3

original_root = os.path.abspath(
    "experiments/cspine_vertebra_mil_maxvit_tiny_d32_stack3/"
    "cspine-vertebra-maxvitt-mil-d32-stack3x96-imageonly-bs2-v1"
)
loaderfix_root = os.path.abspath(
    "experiments/cspine_vertebra_mil_maxvit_tiny_d32_stack3_loaderfix/"
    "cspine-vertebra-maxvitt-mil-d32-stack3x96-imageonly-fold4-loaderfix-v1"
)
cfg.load_pretrained_encoder = os.path.join(
    original_root, "fold0", "checkpoints", "best.ckpt"
)
cfg.mil_checkpoint_by_fold = {
    fold: os.path.join(original_root, f"fold{fold}", "checkpoints", "best.ckpt")
    for fold in range(4)
}
cfg.mil_checkpoint_by_fold[4] = os.path.join(
    loaderfix_root, "fold4", "checkpoints", "best.ckpt"
)
