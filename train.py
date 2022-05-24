import numpy as np
import pandas as pd

from glob import glob
import os, shutil
from tqdm import tqdm
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# PyTorch 
import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.cuda import amp

import timm

#import rasterio
from joblib import Parallel, delayed

from source.config import set_seed

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

from source.dataloader import prepare_loaders, save_batch, plot_batch
from model.model import build_model, load_model

from source.losses import dice_coef, get_criterion, iou_coef, compute_hausdorff_monai, compute_hausdorff_scipy

import hydra
from hydra_plugins.hydra_optuna_sweeper.config import OptunaSweeperConf
from omegaconf import DictConfig, OmegaConf, ListConfig
from source.optimizer import get_optimizer, get_scheduler
#Wandb
import wandb
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("gnzrg25")
    wandb.login(key='0b0a03cb580e75ef44b4dff7f6f16ce9cfa8a290')
    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')

import mlflow

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)
            
def train_one_epoch(cfg, model, optimizer, scheduler, criterion, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    n_accumulate = max(1, 32//cfg.train_config.train_bs)
    
    dataset_size = 0
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train:')
    for step, (images, masks) in pbar:         
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix(epoch=f'{epoch}',train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()
        
        if cfg.train_config.debug and step < 30:
            _imgs  = images.cpu().detach()
            
            _y_pred = (nn.Sigmoid()(y_pred)>0.5).double()
            _y_pred = _y_pred.cpu().detach()
            
            _masks = masks.cpu().detach()
            #_y_pred = torch.mean(torch.stack(_y_pred, dim=0), dim=0).cpu().detach()
            
            plot_batch(imgs=_imgs, pred_msks=_y_pred, gt_msks=_masks, size=5, step = step, epoch = epoch, mode = 'train')
            #save_batch(_imgs, _y_pred, size = 5, step = step, epoch = epoch, mode = 'train')
    return epoch_loss
    
@torch.no_grad()
def valid_one_epoch(cfg, model, dataloader, criterion, device, epoch, optimizer):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    max_distance = np.sqrt(320 ** 2 + 384 ** 2)
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        #val_hausdorff = compute_hausdorff_scipy(masks, y_pred, max_distance).cpu().detach().numpy()
        
        val_scores.append([val_dice, val_jaccard])
        #val_scores.append([val_dice, val_hausdorff])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
        
        if cfg.train_config.debug:
            _imgs  = images.cpu().detach()
            
            #_y_pred = (nn.Sigmoid()(y_pred)>0.5).double()
            _y_pred = y_pred.cpu().detach()
            
            _masks = masks.cpu().detach()
            #_y_pred = torch.mean(torch.stack(_y_pred, dim=0), dim=0).cpu().detach()
            
            plot_batch(imgs=_imgs, pred_msks=_y_pred, gt_msks=_masks, size=5, step = step, epoch = epoch, mode = 'valid')
            #save_batch(_imgs, _y_pred, size = 5, step = step, epoch = epoch, mode = 'valid')
    val_scores  = np.mean(val_scores, axis=0)
    
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores

def run_training(cfg, model, optimizer, scheduler, criterion, device, num_epochs, train_loader, valid_loader, run_log_wandb, fold):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice      = -np.inf
    best_epoch     = -1
    history = defaultdict(list)
    
    # start new run
    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.train_config.comment)
    with mlflow.start_run():
        for epoch in range(1, num_epochs + 1): 
            gc.collect()
            print(f'Epoch {epoch}/{num_epochs}', end='')
            
            # log param
            log_params_from_omegaconf_dict(cfg)
            train_loss = train_one_epoch(cfg, model, optimizer, scheduler, criterion= criterion,
                                            dataloader=train_loader, 
                                            device=device, epoch=epoch)
            
            val_loss, val_scores = valid_one_epoch(cfg, model, valid_loader, criterion,
                                                    device=device, 
                                                    epoch=epoch, optimizer=optimizer)
            #val_dice, val_hausdorff = val_scores
            val_dice, val_jaccard = val_scores
            
            history['Train Loss'].append(train_loss)
            history['Valid Loss'].append(val_loss)
            history['Valid Dice'].append(val_dice)
            history['Valid Jaccard'].append(val_jaccard)
            #history['Valid Hausdorff'].append(val_hausdorff)
            
            # Log the metrics
            wandb.log({"Train Loss": train_loss, 
                    "Valid Loss": val_loss,
                    "Valid Dice": val_dice,
                    "Valid Jaccard": val_jaccard,
                    #"Valid Hausdorff": val_hausdorff,
                    "LR":scheduler.get_last_lr()[0]})
            
            #val_acc = 0.4 * val_dice + 0.6 * val_hausdorff
            val_acc = 0.4 * val_dice + 0.6 * val_jaccard
            
            #Mlflow log
            mlflow.log_metric("Train_loss", train_loss, step=epoch)
            mlflow.log_metric("Train_lr", scheduler.get_last_lr()[0], step=epoch)
            mlflow.log_metric("Val_val_dice", val_dice, step=epoch)
            mlflow.log_metric("Val_val_jaccard", val_jaccard, step=epoch)
            #mlflow.log_metric("Val_val_hausdorff:", val_hausdorff, step=epoch)
            mlflow.log_metric("Val", val_acc, step=epoch)
            
            print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
            
            # deep copy the model
            if val_dice >= best_dice:
                print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
                best_dice    = val_dice
                best_jaccard = val_jaccard
                best_epoch   = epoch
                run_log_wandb.summary["Best Dice"]    = best_dice
                run_log_wandb.summary["Best Jaccard"] = best_jaccard
                run_log_wandb.summary["Best Epoch"]   = best_epoch
                best_model_wts = copy.deepcopy(model.state_dict())

                dirPath = "./run/{}".format(cfg.train_config.comment)
                PATH = f"best_epoch-{fold:02d}.bin"
                torch.save(model.state_dict(), os.path.join(dirPath,PATH))
                # Save a model file from the current directory
                wandb.save(PATH)
                print(f"Model Saved{sr_}")

            last_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"last_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            print(); print()

        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best Score: {:.4f}".format(best_dice))
        
        # load best model weights
        model.load_state_dict(best_model_wts)
    
    return model, history, val_acc

@hydra.main(config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:
    _con = OmegaConf.to_yaml(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed()
    
    # For use original path.
    currPath = os.getcwd()
    os.chdir(currPath)
    print(currPath)
    # org_cwd = hydra.utils.get_original_cwd()
    # print(org_cwd)
    
    dirPath = "./run/{}".format(cfg.train_config.comment)
    if not os.path.isdir(dirPath): 
        os.makedirs(dirPath)
    
    model = build_model(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    criterion = get_criterion(cfg)
    
    print(cfg.train_config.comment)
    
    for fold in cfg.folds.folds:
        print(f'#'*15)
        print(f'### Fold: {fold}')
        print(f'#'*15)
        run_log_wandb = wandb.init(project='uw-maddison-gi-tract', 
                        config={k:v for k, v in dict(cfg).items() if '__' not in k},
                        anonymous=anonymous,
                        name=f"fold-{fold}|dim-{cfg.train_config.img_size[0]}x{cfg.train_config.img_size[1]}|model-{cfg.model.name}|backbone-{cfg.model.backbone}",
                        group=cfg.train_config.comment,
                        )
        train_loader, valid_loader = prepare_loaders(cfg, fold=fold, debug=cfg.train_config.debug)
        model = build_model(cfg)
        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer)
        model, history, val_acc = run_training(cfg, model, optimizer, scheduler, criterion= criterion,
                                    device=cfg.train_config.device,
                                    num_epochs=cfg.train_config.epochs,
                                    train_loader=train_loader,
                                    valid_loader=valid_loader,
                                    run_log_wandb=run_log_wandb,
                                    fold=fold)
        run_log_wandb.finish()
    return val_acc

if __name__ == "__main__":
    train()