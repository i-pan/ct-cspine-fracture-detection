"""Original-style X3D-L pipeline evaluated on the current project folds."""

import copy
import os

from skp.configs.cspine_vertebra_x3d_l_original_repro import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")
cfg.annotations_file = os.path.join(
    work_root,
    "manifests",
    "vertebra_samples_native_depth.parquet",
)
cfg.split_column = "fold"
