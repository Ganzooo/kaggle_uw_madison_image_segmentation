CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[1] model.backbone=efficientnet-b7 train_config.train_bs=16 train_config.valid_bs=16
CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[2] model.backbone=efficientnet-b7 train_config.train_bs=16 train_config.valid_bs=16
CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[3] model.backbone=efficientnet-b7 train_config.train_bs=16 train_config.valid_bs=16
CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[4] model.backbone=efficientnet-b7 train_config.train_bs=16 train_config.valid_bs=16