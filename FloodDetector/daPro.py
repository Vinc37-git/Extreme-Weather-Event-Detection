''' 
@brief
This is a custom module for descprition, visualisation 
and processing of Sentinel 1 SAR Satellite Imagery collected, 
preprocessed and labeled by the Company **cloud2street**: 
the `Sen1Floods11` Dataset.
'''

import copy
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap, ScalarMappable

import rasterio

#import tqdm


class ImageLoader:
    def __init__(self, path_images, path_labels, add_ratio_channel=True, tile_factor=None, nan=None, unsigend_labels=False):
        '''
        An Image Loader Class, optimized for the Sen1Floods11 Dataset. 
        
        Args:
            - `path_images` : The root path of the images.
            - `path_labels` : The root path of the labels.
            - `add_ratio_channel` : A flag to add a composite channel. `TODO:` Change argument name to `add_composite`.
            - `tile_factor` : Divides the images and respective labels in smaller tiles by a factor `tile_factor`
            - `nan` : A numerical number, which serves as a replacement for NaN Values.
            - `unsigned_labels` : Push Label Values to unsigned integer values starting with 0.
        '''
        self.root_images = path_images
        self.root_labels = path_labels
        self.add_ratio_channel = add_ratio_channel
        self.tile_factor = tile_factor
        self.set_nan_to = nan
        self.unsigend_labels = unsigend_labels
        

        
    def csv_split_to_list(self, csv_path, root_images, root_lables):
        '''Reads a csv split file and converts it to a python list containing tuples of format: (Image, labeled Image)'''
        l = []
        with open(csv_path) as f:
            for line in csv.reader(f):
                l.append((root_images + line[0], root_lables + line[1]))
        print(len(l), " Images were found.")
        return l

    
    def load_ds_from_list(self, csv_list):
        '''Loads an Image Dataset by reading a list of paths and returns a 3d numpy array with shape (Image count, height, width, band count)'''
        images = []
        labels = []
        for image_path, label_path in csv_list:
            images.append(rasterio.open(image_path).read().transpose(1,2,0))
            labels.append(rasterio.open(label_path).read().transpose(1,2,0))
        return np.array(images), np.array(labels)
    
    
    def tile_images_and_labels(self, images, labels):
        '''Divides the images and respective labels in smaller tiles by a factor `tile_factor`, which should be set in the constructor.'''
        assert images.shape[1] == images.shape[2], print("Tiling option is only supported for squared images.")
        f = self.tile_factor
        sz = images.shape[1] // f
        tiled_images = []
        tiled_labels = []
        for image, label in zip(images, labels):
            tiled_images.append(np.array([image[x:x+sz,y:y+sz,:] for x in np.arange(f)*sz for y in np.arange(f)*sz]))
            tiled_labels.append(np.array([label[x:x+sz,y:y+sz] for x in np.arange(f)*sz for y in np.arange(f)*sz]))

        tiled_images = np.array(tiled_images).reshape((-1, sz, sz, images.shape[-1]))
        tiled_labels = np.array(tiled_labels).reshape((-1, sz, sz, labels.shape[-1]))
        return tiled_images, tiled_labels

        
    def load_ds(self, csv_path):
        '''Public: Is used to load the Dataset by passing a path to the CSV file.'''
        
        l = self.csv_split_to_list(csv_path, self.root_images, self.root_labels)
        images, labels = self.load_ds_from_list(l)
        
        
        # TODO: This part is a mess. Clean it up!
        
        if self.add_ratio_channel:
            #use_delta_instead_ratio = True
            #if not use_delta_instead_ratio:
            #    ratio_channel = np.clip(np.nan_to_num(images[...,0] / images[...,1]), 0, 2)  # VH / VV Polraisation  # images[...,1] - images[...,0]  # 1 - images[...,0,] / images[...,1]
            #else:
            #    n = 1. # n-th root
            #    ratio_channel = np.abs(images[...,1]**n - images[...,0]**n)**(1./n)
            #    ratio_channel = ratio_channel**2
            #    ratio_channel = np.nan_to_num(ratio_channel, np.min(ratio_channel))# * -1
                
            #gamma = 1
            #beta = 5
            #th = -20
            #ratio_channel = images[...,1] - gamma * (th - images[...,1])**3
            
            
            ratio_channel = np.where(np.nan_to_num(images[...,0]) < -20, -20, np.nan_to_num(images[...,0]))
            ratio_channel = np.where(ratio_channel > 0, 0, ratio_channel)
            
            #ratio_channel = np.where(np.nan_to_num(images[...,1]) < -30, -30, np.nan_to_num(images[...,1]))
            #ratio_channel = np.where(ratio_channel > -10, -10, ratio_channel)

            #ratio_channel = np.where(images[...,1] < -20, images[...,1] - gamma / (np.abs(th-images[...,1])/beta+1), images[...,1] + gamma / (np.abs(th-images[...,1])/beta+1)) 
            images = np.concatenate((images, ratio_channel[...,np.newaxis]), axis=3)
        
        if self.tile_factor is not None and self.tile_factor > 0:
            images, labels = self.tile_images_and_labels(images, labels)
            
        if self.set_nan_to is not None:
            np.nan_to_num(images, copy=False, nan=self.set_nan_to)
            
        if self.unsigend_labels and np.min(labels) < 0:
            labels = labels - np.min(labels)  # TODO: This is not robust. If a loaded dataset (e.g train) consists of -1 labels while another (e.g. test) does not, the shift could be different for labels of the same class.
            
        return images, labels
    
    
def create_mask(pred):
    ''' Creates a 1 channel mask out of a n-channel label prediction by choosing the label with the highest score. '''
    mask = np.argmax(pred, axis=-1)
    mask = mask[..., np.newaxis]
    return mask.astype(np.uint8)


def plot_pixel_dstr_per_ch(images, labels, pos_bound=False):
    ''' Plots the pixel disribution per label for a 3 channel image. TODO: move it into a loop.'''
    fig, ax = plt.subplots(1, 3, figsize=(15,4));
    ax = ax.flatten()
    
    label = [0, 1, 2] if pos_bound else [-1, 0, 1]
    
    ch1 = images[:,:,:,0]
    ax[0].hist(ch1[labels[...,0]==label[0]].flatten(), bins=256, density=True, label="No Data", histtype="step")
    ax[0].hist(ch1[labels[...,0]==label[1]].flatten(), bins=256, density=True, label="No Water", histtype="step")
    ax[0].hist(ch1[labels[...,0]==label[2]].flatten(), bins=256, density=True, label="Water", histtype="step") 
    ax[0].set_title("Channel 1: VV Polarisation", fontweight="bold")
    ax[0].legend()
    ax[0].grid()
    
    ch2 = images[:,:,:,1]
    ax[1].hist(ch2[labels[...,0]==label[0]].flatten(), bins=256, density=True, label="No Data", histtype="step")
    ax[1].hist(ch2[labels[...,0]==label[1]].flatten(), bins=256, density=True, label="No Water", histtype="step")
    ax[1].hist(ch2[labels[...,0]==label[2]].flatten(), bins=256, density=True, label="Water", histtype="step") 
    ax[1].set_title("Channel 2: VH Polarisation", fontweight="bold")
    ax[1].legend()
    ax[1].grid()
    
    ch3 = images[:,:,:,2]
    ax[2].hist(ch3[labels[...,0]==label[0]].flatten(), bins=256, density=True, label="No Data", histtype="step")
    ax[2].hist(ch3[labels[...,0]==label[1]].flatten(), bins=256, density=True, label="No Water", histtype="step")
    ax[2].hist(ch3[labels[...,0]==label[2]].flatten(), bins=256, density=True, label="Water", histtype="step") 
    ax[2].set_title("Channel Ratio", fontweight="bold")
    ax[2].legend()
    ax[2].grid()
    
    numel = labels.size
    fracs = []
    for l in label:
        fracs.append(labels[labels==l].size / numel)
    print(f"Pixel Distribution: No Data: {fracs[0]:.2f}, No Water: {fracs[1]:.2f}, Water: {fracs[2]:.2f}.")
    print("Recommended Weights: ", [0, 1, fracs[1] / fracs[2]], "\n")

    
def plot_rand_imgs(N, imgs, gts, pred=None, id_of_interest=None, pos_bound=True, savepath='plots/default.png'):
    ''' 
    Function to plot random or specific SAR Images, Histograms and Ground Truth Images.
    
    Args:
        - N (int): Number of Images.
        - imgs (list, array): List or Array, which contains the Images.
        - gts (list, array): List or Array, which contains the Ground Truth Images in the same order as imgs.
        - pred (list, array): List or Array, which contains the Predictions in the same order as imgs.
        - pos_bound (bool): GT labels range from [-1 .. 1] or [0 .. 2].
    '''
    
    # create custom colormap for labeled images.
    gt_range = [-0.5, 0.5, 1.5, 2.5] if pos_bound else [-1.5, -0.5, 0.5, 1.5]
    gt_cmap = ListedColormap(['k', 'g', 'b'])
    gt_norm = BoundaryNorm(gt_range, gt_cmap.N, clip=False)
    
    sar_cmap = copy.copy(get_cmap(name="gray"))
    sar_cmap.set_bad(color="red")
    sar_ch1_max, sar_ch1_min = np.nanmax(imgs[:,:,:,0]), np.nanmin(imgs[:,:,:,0])
    sar_ch2_max, sar_ch2_min = np.nanmax(imgs[:,:,:,1]), np.nanmin(imgs[:,:,:,1])
    
    # Number of columns in the subplot
    M = 4 if pred is None else 5
    
    # Pick random or specific indexies
    numbers = np.random.randint(low=0, high=len(imgs), size=N)
    if id_of_interest is not None:
        for i, num in enumerate(id_of_interest):
            numbers[i] = num
    
    # create subplot
    fig, ax = plt.subplots(N, M, figsize=(3 * M, 2.5 * N));
    ax = ax.flatten()

    for i, n in enumerate(numbers):
        # Histograms        
        ax[M * i].hist(imgs[n,:,:,0].flatten(), bins=256, density=False, label="Band 0: VV", histtype="step")
        ax[M * i].hist(imgs[n,:,:,1].flatten(), bins=256, density=False, label="Band 1: VH", histtype="step")
        #ax[M * i].hist(imgs[n,:,:,2].flatten(), bins=256, density=True, label="Band Ratio", histtype="step")
        ax[M * i].set_title("Histogramm of both bands", fontweight="bold")
        ax[M * i].set_xlabel("Backscattering Coefficient [dB]")
        ax[M * i].set_ylabel("Frequency [-]", fontweight="bold")
        ax[M * i].legend()
        
        # SAR Channel 1
        im = ax[M * i + 1].imshow(imgs[n,:,:,0], cmap=sar_cmap, vmin=sar_ch1_min, vmax=sar_ch1_max)
        ax[M * i + 1].set_title(f"Image ID {n}, Band 0: VV", fontweight="bold")
        fig.colorbar(im, ax=ax[M * i + 1], fraction=0.046, pad=0.04, label="Backscattering Coefficient [dB]")
        
        # SAR Channel 2
        im = ax[M * i + 2].imshow(imgs[n,:,:,1], cmap=sar_cmap, vmin=sar_ch2_min, vmax=sar_ch2_max)
        ax[M * i + 2].set_title(f"Image ID {n}, Band 1: VH", fontweight="bold")
        fig.colorbar(im, ax=ax[M * i + 2], fraction=0.046, pad=0.04, label="Backscattering Coefficient [dB]")
        
        # Ground Truth
        im = ax[M * i + 3].imshow(gts[n,:,:,0], cmap=gt_cmap, norm=gt_norm)
        ax[M * i + 3].set_title(f"Image ID {n}, Ground Truth", fontweight="bold")
        fig.colorbar(im, ax=ax[M * i + 3], fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2])

        # Prediction
        if pred is not None:
            im = ax[M * i + 4].imshow(create_mask(pred[n]), cmap=gt_cmap, norm=gt_norm)
            ax[M * i + 4].set_title(f"Image ID {n}, Prediction", fontweight="bold")
            fig.colorbar(im, ax=ax[M * i + 4], fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2])
            
    fig.tight_layout()
    fig.savefig(savepath)
    return fig, ax


def create_sample_weights(gt, weights=[1,1,1]):
    ''' Create sample weight based on a ground truth set. '''
    assert len(weights) == 3, print("Length of weights list must be 3 (number of classes.")
    weights = np.array(weights) / np.sum(weights)
    sample_weights = np.ndarray(gt.shape, dtype='float32')
    sample_weights[gt == 0] = weights[0]
    sample_weights[gt == 1] = weights[1]
    sample_weights[gt == 2] = weights[2]
    return sample_weights


class LinearNorm:
    def __init__(self, v_min=[-40, -30], v_max=[-5, 0]):
        ''' Instanciates a Linear Normalizer. It scales a dataset linear in the range `[v_min ... v_max]. '''
        self.v_min = v_min
        self.v_max = v_max
        
    @classmethod
    def withPercentiles(cls, train_ds, perc_min=5, perc_max=95):
        ''' Constructor overload: Calculate maximum and minimum scaling values using alpha percentiles.'''
        v_min_max = []
        for band in range(train_ds.shape[3]):
            # determine v_min and v_max band-wise by calculating the percentiles.
            v_min_max.append(np.percentile(train_ds[:,:,:,band], q=[perc_min, perc_max]))
        v_min_max = np.array(v_min_max)
        v_min = v_min_max[:,0]
        v_max = v_min_max[:,1]
        print("Min Scaling Limits: ", v_min, " - Max Scaling Limits: ", v_max)
        return cls(v_min, v_max)
        
    def norm(self, data):
        ''' Perform normalization on `data`. '''
        data = data.copy()
        # first clip to vmax and vmin channel-wise, then scale linear
        for band in range(data.shape[3]):
            band_data = data[:,:,:,band]
            # clip
            band_data = np.clip(band_data, a_min=self.v_min[band], a_max=self.v_max[band])
            # scale
            band_data = 2 * (band_data - self.v_min[band]) / (self.v_max[band] - self.v_min[band]) - 1  # (img / (vmax - vmin) + 1 ) *  256
                        
            data[:,:,:,band] = band_data
        return data #preprocess_input(data)
    
    def normPixel(self, pix):
        return [2 * (p - self.v_min[i]) / (self.v_max[i] - self.v_min[i]) - 1 for i, p in enumerate(pix)]
    

class GaussianNorm:
    def __init__(self, ds_train):
        ''' Instanciates a Gaussian Normalizer. It normalizes a dataset in a way that it follows a normal distribution afterwards.'''

        # Calculate the Mean and Std of all SAR values channel-wise.
        self.mean = []
        self.std  = []
        for band in range(ds_train.shape[3]):
            self.mean.append(ds_train[:,:,:,band].flatten().mean())
            self.std.append(ds_train[:,:,:,band].flatten().std())
        print("Normalization Prameter - Mean: ", self.mean, "dB - Std: ", self.std, " dB")
    
    def norm(self, ds):
        ''' Perform normalisation on `ds`.'''
        ds = ds.copy()
        ds = (ds - self.mean) / self.std
        return ds

if __name__ == "__main__":
    pass
    