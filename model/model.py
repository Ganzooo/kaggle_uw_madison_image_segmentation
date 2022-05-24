import segmentation_models_pytorch as smp
from .levit_unet.LeViT_UNet_128s import Build_LeViT_UNet_128s
from .levit_unet.LeViT_UNet_192 import Build_LeViT_UNet_192
from .levit_unet.LeViT_UNet_384 import Build_LeViT_UNet_384
import torch

def build_model(cfg):
    if cfg.model.name == 'Unet':
        model = smp.Unet(
            encoder_name=cfg.model.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=cfg.dataset.num_classes,        # model output channels (number of classes in your dataset)
            activation=None,
            decoder_attention_type="scse",
        )
    elif cfg.model.name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=cfg.model.backbone,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=cfg.dataset.in_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=cfg.dataset.num_classes,        # model output channels (number of classes in your dataset)
            activation=None,
            decoder_attention_type="scse",
        )
    elif cfg.model.name == 'LevitUnet384':
        model = Build_LeViT_UNet_384(
            img_size= cfg.train_config.img_size[0],
            pretrained=True,
            num_classes=cfg.dataset.num_classes,        # model output channels (number of classes in your dataset)
        )
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(cfg.train_config.device)
    return model

def load_model(cfg, path):
    model = build_model(cfg)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == "__main__":
    cfg = 'None'
    modelTrain = build_model(cfg)
    print(modelTrain)