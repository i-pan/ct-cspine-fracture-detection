from copy import deepcopy

from skp.configs.cspine_vertebra_mil_effnetv2_s_d48 import cfg as base_cfg


cfg = deepcopy(base_cfg)
cfg.project = "cspine_vertebra_mil_effnetv2_s_d48_bs4"
cfg.batch_size = 4
cfg.val_batch_size = 4
# A local 4/GPU batch contains at most 4 * 48 valid frames.
cfg.encoder_chunk_size = 192
