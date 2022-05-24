#CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[0]
#CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[1]
#CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[2]
#CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[3]
#CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[4]
CUDA_VISIBLE_DEVICES=0 python3 unet_train_new.py folds.folds=[1] model.backbone=efficientnet-b7 train_config.train_bs=16 train_config.valid_bs=16 train_config.epochs=20 model.backbone="resnet152"