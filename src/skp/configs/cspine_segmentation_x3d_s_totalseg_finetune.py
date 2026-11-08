import copy
from pathlib import Path

from skp.configs.cspine_segmentation_x3d_s import cfg as base_cfg


cfg = copy.deepcopy(base_cfg)
project_root = Path(__file__).resolve().parents[3]
pretrain_root = (
    project_root
    / "experiments"
    / "cspine_segmentation_x3d_s_totalseg_pretrain"
    / "cspine-seg-x3ds-ts-pretrain-v1"
)

cfg.project = "cspine_segmentation_x3d_s_totalseg_finetune"
cfg.pretrained = False
cfg.load_pretrained_model = str(
    pretrain_root / "fold0" / "checkpoints" / "best.ckpt"
)
cfg.num_epochs = 50
cfg.optimizer_params = {"lr": 1e-4, "weight_decay": 1e-2}
cfg.scheduler_params = {"pct_start": 0.05, "init_lr": 0.0, "final_lr": 1e-6}
