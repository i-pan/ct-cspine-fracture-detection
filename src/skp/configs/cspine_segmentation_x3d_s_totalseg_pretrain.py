import copy
import os

from skp.configs.cspine_segmentation_x3d_s import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
work_root = os.environ.get("CSPINE_WORK_ROOT", "./data/processed")

cfg.project = "cspine_segmentation_x3d_s_totalseg_pretrain"
cfg.annotations_file = os.path.join(
    work_root, "manifests", "totalsegmentator_samples_qc.parquet"
)
cfg.validation_annotations_file = os.path.join(
    work_root, "manifests", "segmentation_samples.parquet"
)
cfg.num_epochs = 20
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
