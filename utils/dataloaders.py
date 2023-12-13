from math import floor, ceil

import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).type(torch.float)
            
        if self.data.dim() == 2:
            self.data = self.data[:, np.newaxis]
        elif self.data.dim() != 3:
            print('Data dimensions should be [B, C, W] or [B, W]')
                
    def getparams(self):
        data_mean = torch.mean(self.data)
        data_std = torch.std(self.data)
        return data_mean, data_std
    
    def getimgwidth(self):
        img = self.__getitem__(0)
        return img.shape[-1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        
        return img
        

def create_nm_loader(n_data, split=0.8, batch_size=32):
    """Creates pytorch dataloaders for training the noise model.
    
    Parameters
    ----------
    n_data : numpy ndarray
        The noise image data.
    split : Float, optional
        Percent of data to go into the training set, remaining
        data will go into the validation set. The default is 0.8.
    batch_size : int, optional
        Size of batches. The default is 32.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the training set.
    val_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the validation set.
    noise_mean : float
        Mean of the noise data, used to normalise.
    noise_std : float
        Standard deviation of the noise data, used to normalise.

    """
    dataset = Dataset(n_data)
    
    train_set, val_set = torch.utils.data.random_split(dataset, [floor(len(dataset)*split), ceil(len(dataset)*(1-split))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    noise_mean, noise_std = dataset.getparams()
    return train_loader, val_loader, noise_mean, noise_std


def create_dn_loader(x_data, split=0.9, batch_size=32):
    """Creates pytorch dataloaders for training the denoising VAE.
    
    Parameters
    ----------
    x_data : numpy ndarray
        The noisy image data.
    split : Float, optional
        Percent of data to go into the training set, remaining
        data will go into the validation set. The default is 0.8.
    batch_size : int, optional
        Size of batches. The default is 32.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the training set.
    val_loader : torch.utils.data.DataLoader class
        Pytorch dataloader of the validation set.
    img_shape : list of ints
        Height and width of the images, used to prepare
        the VAE.
    data_mean : float
        Mean of the noisy image data, used to normalise.
    data_std : float
        Standard deviation of the noisy image data, used to normalise.

    """
    dataset = Dataset(x_data)
    
    img_shape = dataset.getimgwidth()
    
    data_mean, data_std = dataset.getparams()
    
    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset)*split), round(len(dataset)*(1-split))])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader, img_shape, data_mean, data_std
