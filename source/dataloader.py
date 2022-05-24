import pandas as pd
from glob import glob
import numpy as np

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#from config import CFG

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

import os

#Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def prepare_loaders(cfg, fold, debug=False):
    df = prepare_data_to_df(cfg)
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    # if debug:
    #     train_df = train_df.head(32*5).query("empty==0")
    #     valid_df = valid_df.head(32*3).query("empty==0")

    data_transforms = {
        "train": A.Compose([
            A.Resize(*cfg.train_config.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                #A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GaussNoise(var_limit=(0.001, 0.01), mean=0, per_channel=False, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=cfg.train_config.img_size[0]//20, max_width=cfg.train_config.img_size[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
        "valid": A.Compose([
            A.Resize(*cfg.train_config.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
    }

    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_config.train_bs, 
                              num_workers=cfg.dataset.num_workers, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.train_config.valid_bs, 
                              num_workers=cfg.dataset.num_workers, shuffle=False, pin_memory=True)
    return train_loader, valid_loader

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.msk_paths  = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = self.load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = self.load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)

    def load_img(self, path):
        img = np.load(path)
        img = img.astype('float32') # original is uint16
        mx = np.max(img)
        if mx:
            img/=mx # scale image to [0, 1]
        return img

    def load_msk(self, path):
        msk = np.load(path)
        msk = msk.astype('float32')
        msk/=255.0
        return msk

def id2mask(id_):
    idf = self.df[self.df['id']==id_]
    wh = idf[['height','width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
        cdf = idf[idf['class']==class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask

def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0,0),(0,0),(1,0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask

def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     img = clahe.apply(img)
#     plt.figure(figsize=(10,10))
    #plt.imshow(img, cmap='bone')
    cv2.imwrite('./ped_img.jpg', img)
    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
        
    cv2.imwrite('./mask_img.jpg', mask)
    plt.axis('off')

def prepare_data_to_df(cfg):
    path_df = pd.DataFrame(glob('/dataset/uw-madison-gi-tract-segmentation/data2d/images/images/*'), columns=['image_path'])
    path_df['mask_path'] = path_df.image_path.str.replace('image','mask')
    path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy',''))
    #path_df.head()

    df = pd.read_csv('/dataset/uw-madison-gi-tract-segmentation/uwmgi-mask-dataset/train.csv')
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len) # length of each rle mask

    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
    df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

    df['image_path'] = df.mask_path.str.replace('/png/','/np').str.replace('.png','.npy')
    #df.head()

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len==0) # empty masks

    df = df.drop(columns=['image_path','mask_path'])
    df = df.merge(path_df, on=['id'])
    #df.head()

    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)
    #df.head()

    df['empty'].value_counts().plot.bar()
    
    skf = StratifiedGroupKFold(n_splits=cfg.folds.n_fold, shuffle=True, random_state=cfg.train_config.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold
        #display(df.groupby(['fold','empty'])['id'].count())
    return df



# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def show_img(img, f, axes, i, j, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #plt.imshow(img, cmap='bone')
    axes[i,j].imshow(img, cmap='bone')
    
    if mask is not None:
        ## plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        #plt.imshow(mask, alpha=0.5)
        axes[i,j].imshow(mask, alpha=0.5)
        # handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        # labels = ["Large Bowel", "Small Bowel", "Stomach"]
        # #plt.legend(handles,labels)
        # axes[i,j].legend(handles,labels)
    #plt.axis('off')
    axes[i,j].axis('off')
    
       
def plot_batch(imgs, pred_msks, gt_msks, size=3, step=1, epoch=0, mode='train'):
    f, axes = plt.subplots(2, 5, figsize=(50,50),constrained_layout=True )
    #f.set_size_inches((12, 12))
    #plt.figure(figsize=(50, 50))
    plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
    #plt.subplot(2, 5, 5)
    
    handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
    labels = ["Large Bowel", "Small Bowel", "Stomach"]
    f.legend(handles,labels)
        
    for idx in range(size):
        #plt.subplot(1, size, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        _pred_msks = pred_msks[idx,].permute((1, 2, 0)).numpy()*255.0
        _pred_msks = _pred_msks.astype('uint8')
        show_img(img, f, axes, 0, idx, _pred_msks)
    f.tight_layout()
     
    for idx in range(size):
        #plt.subplot(2, size, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        
        _gt_msks = gt_msks[idx,].permute((1, 2, 0)).numpy()*255.0
        _gt_msks = _gt_msks.astype('uint8')
        
        show_img(img, f, axes, 1, idx, _gt_msks)
        
    f.tight_layout()
    #f.show()
    
    currPath = os.getcwd()
    os.chdir(currPath)
    #print(currPath)
    
    file_name = "pred_{}_{}_{}.png".format(mode, step, idx)
    dir = "{}/{}/".format(mode,epoch)
    fdir = os.path.join(currPath,dir)   
    if not os.path.exists(fdir):
        os.makedirs(fdir) 
    f.savefig(fdir + file_name)

def save_img(filepath, img, color_domain='rgb'):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    #if color_domain == 'ycbcr':
    #    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR))
    #else:
    cv2.imwrite(filepath,img)
    
def save_batch(imgs, msks, size=3, step=1, epoch=0, mode='train'):
    for idx in range(size):
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy()*255.0
        _img = img + msk
        file_name = "./{}/{}/pred_{}_{}.jpg".format(mode,epoch,step,idx) 
        save_img(file_name,_img)    
    #show_img(img, msk)
    #plt.tight_layout()
    #plt.show()
#plot_batch(imgs, msks, size=5)

if __name__ == "__main__":
    train_loader, valid_loader = prepare_loaders(fold=0, debug=True)

    imgs, msks = next(iter(train_loader))
    print("img size:", imgs.size())
    print("mask size:", msks.size())
    #plot_batch(imgs, msks, size=5)
