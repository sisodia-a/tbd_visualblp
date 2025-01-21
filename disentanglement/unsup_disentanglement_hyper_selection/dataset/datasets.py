import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
from skimage.io import imread
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from shutil import copyfile

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1

def get_img_size(dataset):
    """Return the correct image size."""
    return (1, 128, 128) 

def get_background(dataset):
    """Return the image background color."""
    return COLOUR_WHITE

def split_dataset(model_name):
    """Split the dataset."""

    root=os.path.join(DIR, '../data/cars/') 
    data = os.path.join(root, "cars_original.npz")

    dataset_zip = np.load(data)
    cars = dataset_zip['cars']
 
    make = dataset_zip['make']
    makemodel = dataset_zip['makemodel']
    color = dataset_zip['color']
    firm = dataset_zip['firm']
    region = dataset_zip['region']
    price = dataset_zip['price']
    hp = dataset_zip['hp']
    mpg = dataset_zip['mpg']
    mpd = dataset_zip['mpd']
    filenames = dataset_zip['filenames']
    in_uk_blp = dataset_zip['in_uk_blp']
    hpwt = dataset_zip['hpwt']
    space = dataset_zip['space']
    wt = dataset_zip['wt']
    length = dataset_zip['length']
    wid = dataset_zip['wid']
    ht = dataset_zip['ht']
    wb = dataset_zip['wb']
    xi_fe = dataset_zip['xi_fe']   
    shares = dataset_zip['shares']
    wph = dataset_zip['wph']

    sequence = np.arange(0,filenames.shape[0])
    modelname = np.argmax(makemodel,axis=1)
    df = pd.DataFrame(data=np.column_stack((sequence,modelname,filenames)),columns=['seq','model','file'])
    df['model'] = df['model'].str.encode('ascii', 'ignore').str.decode('ascii')

    df_mod = df.groupby(['model'])["seq"].count().reset_index(name="count")
    r = np.random.uniform(size=df_mod.shape[0])
    r = np.where(r>=0.9,1,0) ## split ratio
    df_mod['r'] = r.tolist()
    result = pd.merge(df, df_mod, on="model")
    train_idx = result[result['r']==0]
    valid_idx = result[result['r']==1]
    train_idx = train_idx['seq'].to_numpy()
    valid_idx = valid_idx['seq'].to_numpy()
    train_idx = train_idx.astype(np.int)
    valid_idx = valid_idx.astype(np.int)

    # print(make[train_idx,].shape)
    # print(firm[train_idx,].shape)
    # print(region[train_idx,].shape)

    # print(make[valid_idx,].shape)
    # print(firm[valid_idx,].shape)
    # print(region[valid_idx,].shape)

    np.savez( os.path.join(DIR, "../results",model_name,"cars_train.npz"),cars=cars[train_idx,:,:,],make=make[train_idx,],makemodel=makemodel[train_idx,],color=color[train_idx,],firm=firm[train_idx,],region=region[train_idx,],price=price[train_idx],hp=hp[train_idx],mpg=mpg[train_idx],mpd=mpd[train_idx],filenames=filenames[train_idx],in_uk_blp=in_uk_blp[train_idx],hpwt=hpwt[train_idx],space=space[train_idx],wt=wt[train_idx],length=length[train_idx],wid=wid[train_idx],ht=ht[train_idx],wb=wb[train_idx],xi_fe=xi_fe[train_idx],shares=shares[train_idx],wph=wph[train_idx])
    np.savez( os.path.join(DIR, "../results",model_name,"cars_validation.npz"),cars=cars[valid_idx,:,:,],make=make[valid_idx,],makemodel=makemodel[valid_idx,],color=color[valid_idx,],firm=firm[valid_idx,],region=region[valid_idx,],price=price[valid_idx],hp=hp[valid_idx],mpg=mpg[valid_idx],mpd=mpd[valid_idx],filenames=filenames[valid_idx],in_uk_blp=in_uk_blp[valid_idx],hpwt=hpwt[valid_idx],space=space[valid_idx],wt=wt[valid_idx],length=length[valid_idx],wid=wid[valid_idx],ht=ht[valid_idx],wb=wb[valid_idx],xi_fe=xi_fe[valid_idx],shares=shares[valid_idx],wph=wph[valid_idx])

    copyfile(os.path.join(root, "cars_test.npz"),os.path.join(DIR, "../results",model_name,"cars_test.npz"))
    
    return 0

def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128,eval_batchsize=10000,model_name="temp",sup_signal1="xi_fe",sup_signal2="price",sup_signal3="mpg",sup_signal4="hpwt",sup_signal5="space",logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader
    Parameters
    ----------
    dataset :   Name of the dataset to load
    root : str  Path to the dataset root. If `None` uses the default one.
    kwargs :    Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available

    temp = split_dataset(model_name)

    Train_Dataset = Cars(split="train",model_name=model_name,sup_signal1=sup_signal1,sup_signal2=sup_signal2,sup_signal3=sup_signal3,sup_signal4=sup_signal4,sup_signal5=sup_signal5,logger=logger)
    Validation_Dataset = Cars(split="validation",model_name=model_name,sup_signal1=sup_signal1,sup_signal2=sup_signal2,sup_signal3=sup_signal3,sup_signal4=sup_signal4,sup_signal5=sup_signal5,logger=logger)
    Test_Dataset = Cars(split="test",model_name=model_name,sup_signal1=sup_signal1,sup_signal2=sup_signal2,sup_signal3=sup_signal3,sup_signal4=sup_signal4,sup_signal5=sup_signal5,logger=logger)

    train_loader = DataLoader(Train_Dataset,batch_size=batch_size,shuffle=True,pin_memory=pin_memory,**kwargs)
    validation_loader = DataLoader(Validation_Dataset,batch_size=eval_batchsize,shuffle=True,pin_memory=pin_memory,**kwargs)
    train_loader_all = DataLoader(Train_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs) 
    train_loader_one = DataLoader(Train_Dataset,batch_size=1,shuffle=True,pin_memory=pin_memory,**kwargs)
    test_loader = DataLoader(Test_Dataset,batch_size=eval_batchsize,shuffle=False,pin_memory=pin_memory,**kwargs)
    
    return train_loader, validation_loader,  train_loader_all, train_loader_one, test_loader

class Cars(Dataset):
    """
    """
    files = {"train": "cars_train.npz", "validation": "cars_validation.npz", "test": "cars_test.npz", "all":"cars_original.npz"}
    img_size = (1, 128, 128)
    background_color = COLOUR_WHITE
    def __init__(self, root=os.path.join(DIR, '../results/'), transforms_list=[transforms.ToPILImage(),transforms.Resize((128)),transforms.Grayscale(1),transforms.ToTensor()], logger=logging.getLogger(__name__), split="train",model_name="temp",sup_signal1="xi_fe",sup_signal2="price",sup_signal3="mpg",sup_signal4="hpwt",sup_signal5="space",**kwargs):
        self.model_name = model_name
        self.sup_signal1 = sup_signal1
        self.sup_signal2 = sup_signal2
        self.sup_signal3 = sup_signal3
        self.sup_signal4 = sup_signal4
        self.sup_signal5 = sup_signal5
        self.data = os.path.join(DIR,'../results/',self.model_name, type(self).files[split])

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        dataset_zip = np.load(self.data)
        self.imgs = dataset_zip['cars']
        self.make = dataset_zip['make']
        self.makemodel = dataset_zip['makemodel']
        self.color = dataset_zip['color']
        self.firm = dataset_zip['firm']
        self.region = dataset_zip['region']
        self.price = dataset_zip['price']
        self.price=(self.price-np.mean(self.price))/np.std(self.price)
        self.hp = dataset_zip['hp']
        self.hp=(self.hp-np.mean(self.hp))/np.std(self.hp)
        self.mpg = dataset_zip['mpg']
        self.mpg=(self.mpg-np.mean(self.mpg))/np.std(self.mpg)
        self.mpd = dataset_zip['mpd']
        self.mpd=(self.mpd-np.mean(self.mpd))/np.std(self.mpd)
        self.hpwt = dataset_zip['hpwt']
        self.hpwt = (self.hpwt-np.mean(self.hpwt))/np.std(self.hpwt)
        self.space = dataset_zip['space']
        self.space = (self.space-np.mean(self.space))/np.std(self.space)
        self.wt = dataset_zip['wt']
        self.wt = (self.wt-np.mean(self.wt))/np.std(self.wt)
        self.length = dataset_zip['length']
        self.length = (self.length-np.mean(self.length))/np.std(self.length)
        self.wid = dataset_zip['wid']
        self.wid = (self.wid-np.mean(self.wid))/np.std(self.wid)
        self.ht = dataset_zip['ht']
        self.ht = (self.ht-np.mean(self.ht))/np.std(self.ht)
        self.wb = dataset_zip['wb']
        self.wb = (self.wb-np.mean(self.wb))/np.std(self.wb)
        self.xi_fe = dataset_zip['xi_fe']
        self.xi_fe = (self.xi_fe-np.mean(self.xi_fe))/np.std(self.xi_fe)
        self.shares = dataset_zip['shares']
        self.shares = (self.shares-np.mean(self.shares))/np.std(self.shares)
        self.wph = dataset_zip['wph']
        self.wph = (self.wph-np.mean(self.wph))/np.std(self.wph)

        # self.modelname = dataset_zip['modelname']
        self.filenames = dataset_zip['filenames']

        if self.sup_signal5 == 'price':
           self.wtp5 = self.price
        elif self.sup_signal5 == 'hp':
           self.wtp5 = self.hp
        elif self.sup_signal5 == 'wt':
           self.wtp5 = self.wt
        elif self.sup_signal5 == 'hpwt':
           self.wtp5 = self.hpwt
        elif self.sup_signal5 == 'mpg':
           self.wtp5 = self.mpg
        elif self.sup_signal5 == 'mpd':
           self.wtp5 = self.mpd
        elif self.sup_signal5 == 'space':
           self.wtp5 = self.space
        elif self.sup_signal5 == 'length':
           self.wtp5 = self.length
        elif self.sup_signal5 == 'wid':
           self.wtp5 = self.wid
        elif self.sup_signal5 == 'ht':
           self.wtp5 = self.ht
        elif self.sup_signal5 == 'wb':
           self.wtp5 = self.wb
        elif self.sup_signal5 == 'wph':
           self.wtp5 = self.wph
        elif self.sup_signal5 == 'xi_fe':
           self.wtp5 = self.xi_fe
        elif self.sup_signal5 == 'shares':
           self.wtp5 = self.shares

        if self.sup_signal4 == 'price':
           self.wtp4 = self.price
        elif self.sup_signal4 == 'hp':
           self.wtp4 = self.hp
        elif self.sup_signal4 == 'wt':
           self.wtp4 = self.wt
        elif self.sup_signal4 == 'hpwt':
           self.wtp4 = self.hpwt
        elif self.sup_signal4 == 'mpg':
           self.wtp4 = self.mpg
        elif self.sup_signal4 == 'mpd':
           self.wtp4 = self.mpd
        elif self.sup_signal4 == 'space':
           self.wtp4 = self.space
        elif self.sup_signal4 == 'length':
           self.wtp4 = self.length
        elif self.sup_signal4 == 'wid':
           self.wtp4 = self.wid
        elif self.sup_signal4 == 'ht':
           self.wtp4 = self.ht
        elif self.sup_signal4 == 'wb':
           self.wtp4 = self.wb
        elif self.sup_signal4 == 'wph':
           self.wtp4 = self.wph
        elif self.sup_signal4 == 'xi_fe':
           self.wtp4 = self.xi_fe
        elif self.sup_signal4 == 'shares':
           self.wtp4 = self.shares

        if self.sup_signal3 == 'price':
           self.wtp3 = self.price
        elif self.sup_signal3 == 'hp':
           self.wtp3 = self.hp
        elif self.sup_signal3 == 'wt':
           self.wtp3 = self.wt
        elif self.sup_signal3 == 'hpwt':
           self.wtp3 = self.hpwt
        elif self.sup_signal3 == 'mpg':
           self.wtp3 = self.mpg
        elif self.sup_signal3 == 'mpd':
           self.wtp3 = self.mpd
        elif self.sup_signal3 == 'space':
           self.wtp3 = self.space
        elif self.sup_signal3 == 'length':
           self.wtp3 = self.length
        elif self.sup_signal3 == 'wid':
           self.wtp3 = self.wid
        elif self.sup_signal3 == 'ht':
           self.wtp3 = self.ht
        elif self.sup_signal3 == 'wb':
           self.wtp3 = self.wb
        elif self.sup_signal3 == 'wph':
           self.wtp3 = self.wph
        elif self.sup_signal3 == 'xi_fe':
           self.wtp3 = self.xi_fe
        elif self.sup_signal3 == 'shares':
           self.wtp3 = self.shares

        if self.sup_signal2 == 'price':
           self.wtp2 = self.price
        elif self.sup_signal2 == 'hp':
           self.wtp2 = self.hp
        elif self.sup_signal2 == 'wt':
           self.wtp2 = self.wt
        elif self.sup_signal2 == 'hpwt':
           self.wtp2 = self.hpwt
        elif self.sup_signal2 == 'mpg':
           self.wtp2 = self.mpg
        elif self.sup_signal2 == 'mpd':
           self.wtp2 = self.mpd
        elif self.sup_signal2 == 'space':
           self.wtp2 = self.space
        elif self.sup_signal2 == 'length':
           self.wtp2 = self.length
        elif self.sup_signal2 == 'wid':
           self.wtp2 = self.wid
        elif self.sup_signal2 == 'ht':
           self.wtp2 = self.ht
        elif self.sup_signal2 == 'wb':
           self.wtp2 = self.wb
        elif self.sup_signal2 == 'wph':
           self.wtp2 = self.wph
        elif self.sup_signal2 == 'xi_fe':
           self.wtp2 = self.xi_fe
        elif self.sup_signal2 == 'shares':
           self.wtp2 = self.shares

        if self.sup_signal1 == 'price':
           self.wtp1 = self.price
        elif self.sup_signal1 == 'hp':
           self.wtp1 = self.hp
        elif self.sup_signal1 == 'wt':
           self.wtp1 = self.wt
        elif self.sup_signal1 == 'hpwt':
           self.wtp1 = self.hpwt
        elif self.sup_signal1 == 'mpg':
           self.wtp1 = self.mpg
        elif self.sup_signal1 == 'mpd':
           self.wtp1 = self.mpd
        elif self.sup_signal1 == 'space':
           self.wtp1 = self.space
        elif self.sup_signal1 == 'length':
           self.wtp1 = self.length
        elif self.sup_signal1 == 'wid':
           self.wtp1 = self.wid
        elif self.sup_signal1 == 'ht':
           self.wtp1 = self.ht
        elif self.sup_signal1 == 'wb':
           self.wtp1 = self.wb
        elif self.sup_signal1 == 'wph':
           self.wtp1 = self.wph
        elif self.sup_signal1 == 'xi_fe':
           self.wtp1 = self.xi_fe
        elif self.sup_signal1 == 'shares':
           self.wtp1 = self.shares
        elif self.sup_signal1 == 'make':
           self.wtp1 = self.make
           self.wtp1 = np.argmax(self.wtp1,axis=1)
        elif self.sup_signal1 == 'region':
           self.wtp1 = self.region
           self.wtp1 = np.argmax(self.wtp1,axis=1)
        elif self.sup_signal1 == 'color':
           self.wtp1 = self.color
           self.wtp1 = np.argmax(self.wtp1,axis=1)


    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        img = self.transforms(self.imgs[idx])
        wtp1 = self.wtp1[idx]
        wtp2 = self.wtp2[idx]
        wtp3 = self.wtp3[idx]
        wtp4 = self.wtp4[idx]
        wtp5 = self.wtp5[idx]
        make = self.make[idx]
        makemodel = self.makemodel[idx]
        color = self.color[idx]
        firm = self.firm[idx]
        region = self.region[idx]
        price = self.price[idx]
        hp = self.hp[idx]
        mpg = self.mpg[idx]
        mpd = self.mpd[idx]
        filenames = self.filenames[idx]
        hpwt = self.hpwt[idx]
        space = self.space[idx]
        wt = self.wt[idx]
        length = self.length[idx]
        wid = self.wid[idx]
        ht = self.ht[idx]
        wb = self.wb[idx]
        xi_fe = self.xi_fe[idx]
        shares = self.shares[idx]
        wph = self.wph[idx]
        return img, 0, wtp1, wtp2, wtp3, wtp4, wtp5, make, makemodel, color, firm, region, price, hp, mpg, mpd, filenames, hpwt, space, wt, length, wid, ht, wb, xi_fe, shares, wph

# HELPERS
def preprocess(root, size=(128, 128), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
