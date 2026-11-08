from skp.configs.cspine_vertebra_mil_base import build_cfg


cfg = build_cfg(
    project="cspine_vertebra_fracture_classification",
    backbone="csn_r26",
)
cfg.model = "classification.cspine_x3d_mil"
cfg.volume_3d_input = True
cfg.num_input_channels = 2
cfg.include_last_block = False
cfg.x3d_feature_dim = 2048
cfg.dim0_strides = [1, 1, 1, 1, 1]
cfg.output_stride = 32
cfg.mil_pool_temperature = 0.5
cfg.mil_pool_normalize = True
cfg.normalization = "0_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.loss = "classification.BCEWithLogitsLoss"
cfg.loss_params = {}
cfg.num_epochs = 8
cfg.scheduler_num_epochs = 12
