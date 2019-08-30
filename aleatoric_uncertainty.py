from keras.models import load_model
from glob import glob
import keras
import numpy as np
from losses import *
import random
from keras.models import Model
from extract_patches import Pipeline
from scipy.misc import imresize
from keras.utils import np_utils
import SimpleITK as sitk
import pdb
import matplotlib.pyplot as plt
import os
from scipy.ndimage.measurements import label
import cv2 
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import matplotlib.gridspec as gridspec
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# from evaluation_metrics import *

path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path=glob('/home/parth/Interpretable_ML/BraTS_2018/train/**')
np.random.seed(2022)
np.random.shuffle(test_path)


def normalize_scheme(slice_not):
    '''
        normalizes each slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
    '''
    normed_slices = np.zeros(( 4,155, 240, 240))
    for slice_ix in range(4):
        normed_slices[slice_ix] = slice_not[slice_ix]
        for mode_ix in range(155):
            normed_slices[slice_ix][mode_ix] = _normalize(slice_not[slice_ix][mode_ix])

    return normed_slices    


def _normalize(slice):

    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    image_nonzero = slice[np.nonzero(slice)]

    if np.std(slice)==0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp==tmp.min()]=-9
        return tmp

def load_vol(filepath_image, model_type, slice_):

    '''
    segment the input volume
    INPUT   (1) str 'filepath_image': filepath of the volume to predict 
            (2) bool 'show': True to ,
    OUTPUt  (1) np array of the predicted volume
            (2) np array of the corresping ground truth
    '''

    #read the volume
    flair = glob( filepath_image + '/*_flair.nii.gz')
    t2 = glob( filepath_image + '/*_t2.nii.gz')
    gt = glob( filepath_image + '/*_seg.nii.gz')
    t1s = glob( filepath_image + '/*_t1.nii.gz')
    t1c = glob( filepath_image + '/*_t1ce.nii.gz')
    
    t1=[scan for scan in t1s if scan not in t1c]
    if (len(flair)+len(t2)+len(gt)+len(t1)+len(t1c))<5:
        print("there is a problem here!!! the problem lies in this patient :")
    scans_test = [flair[0], t1[0], t1c[0], t2[0], gt[0]]
    test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i])) for i in range(len(scans_test))]


    test_im=np.array(test_im).astype(np.float32)
    test_image = test_im[0:4]
    gt=test_im[-1]
    gt[gt==4]=3

    #normalize each slice following the same scheme used for training
    test_image = normalize_scheme(test_image)

    #transform teh data to channels_last keras format
    test_image = test_image.swapaxes(0,1)
    test_image=np.transpose(test_image,(0,2,3,1))
   
    test_image, gt = np.array(test_image[slice_]), np.array(gt[slice_])
    if model_type == 'dense':
        npad = ((8, 8), (8, 8), (0, 0))
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=-9)
        npad = ((8, 8), (8, 8))
        gt = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)

    return test_image, gt

class Test_Time_Augmentation():
    
    def __init__(self):
        pass
        # self.aug = iaa.SomeOf(2, [
        #                 iaa.Affine(
        #                 rotate=),
        #                 #iaa.AdditiveGaussianNoise(scale=0.3 * np.ptp(test_image) - 9),
        #                 iaa.Noop(),
        #                 iaa.MotionBlur(k=3, angle = [-1, 1])
        #             ], random_order=True)
        
    def predict_aleatoric(self, model, test_image, iterations=1000):
        
        predictions = []
        
        for i in range(iterations):

            rot = 1
            flip = random.choice([0, 1])

            aug = iaa.SomeOf(1, [
                        #iaa.Fliplr(flip),
                        iaa.Affine(
                        rotate=(rot, rot))
                        #iaa.AdditiveGaussianNoise(scale=0.3 * np.ptp(test_image) - 9),
                        #iaa.Noop()
                        

                    ], random_order=False)
            #npad = ((0, 0), (16, 16), (16, 16), (0, 0))
            #test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)
            aug_image = aug.augment_images(test_image)

            aug_i = iaa.SomeOf(1, [
                        iaa.Affine(
                        rotate=(-rot, -rot))
                        #iaa.AdditiveGaussianNoise(scale=0.3 * np.ptp(test_image) - 9),
                        #iaa.Noop(),
                        #iaa.Fliplr(flip)
                    ], random_order=False)

            prediction = model.predict(aug_image.reshape((1, 256, 256, 4)))
            prediction = aug_i.augment_images(prediction)
            predictions.append(prediction)
            
            plt.imshow(np.argmax(predictions[-1], axis = -1).reshape((256, 256)))
            plt.show()
            
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis = 0)
        
        var = np.var(predictions, axis = 0)
        
        print(mean.shape)
                
        # plt.imshow(np.argmax(mean, axis = -1).reshape((240, 240)), vmin = 0., vmax = 3.)
        # plt.show()
        # plt.figure(figsize = (8, 8))
        # plt.imshow(np.mean(var[:, :, :, 1:], axis = -1).reshape((240, 240)))
        # plt.colorbar()
        # plt.show()

        return(mean, var)

    def predict_epistemic(self, model, test_image, iterations=1000):
        
        predictions = []
        
        for i in range(iterations):

            #aug_image = self.aug.augment_images(test_image)
            predictions.append(model.predict(test_image.reshape((1, 256, 256, 4))))

            
        predictions = np.array(predictions)
        
        mean = np.mean(predictions, axis = 0)
        
        var = np.var(predictions, axis = 0)
        
        print(mean.shape)
                
        # plt.imshow(np.argmax(mean, axis = -1).reshape((240, 240)), vmin = 0., vmax = 3.)
        # plt.show()
        # plt.figure(figsize = (8, 8))
        # plt.imshow(np.mean(var[:, :, :, 1:], axis = -1).reshape((240, 240)))
        # plt.colorbar()
        # plt.show()

        return(mean, var)


if __name__ == '__main__':

    model_epistemic = load_model('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/U_densenet/densedrop.h5', 
        custom_objects={'gen_dice_loss':gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
    model_epistemic.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/U_densenet/densedrop_lrsch.hdf5', by_name = True)

    # model_aleatoric = load_model('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/U_densenet/densenet121.h5', 
    #     custom_objects={'gen_dice_loss':gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
    # model_aleatoric.load_weights('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/U_densenet/dense_lrsch.hdf5', by_name = True)

    for volume in range(len(test_path)):

        test_image, gt = load_vol(test_path[volume], 'dense', 78)
        print(test_image.shape)

        # predict classes of each pixel based on the model
        # prediction = model.predict(test_image.reshape((1, 240, 240, 4)), batch_size=1)   
        
        # prediction_not_reshaped = prediction.copy()
        # prediction = np.argmax(prediction, axis=-1)
        # prediction=prediction.astype(np.uint8)
        # #reconstruct the initial target values .i.e. 0,1,2,4 for prediction and ground truth
        # prediction[prediction==3]=4
        # plt.imshow(prediction.reshape((240, 240)))
        # plt.show()

        D = Test_Time_Augmentation()

        imsize = (256, 256)

        mean_ep, var_ep = D.predict_epistemic(model_epistemic, test_image, iterations = 100)
        mean_ep = np.argmax(mean_ep, axis = -1)
        mean_ep = np.ma.masked_where(mean_ep == 0, mean_ep)
        gt = np.ma.masked_where(gt == 0, gt)
        var_ep = np.mean(var_ep[:, :, :, 1:], axis = -1)
        #var_ep = np.ma.masked_where(mean_ep == 0, var_ep)

        #mean_al, var_al = D.predict_aleatoric(model_aleatoric, test_image, iterations = 200)   
                # Choose colormap
        cmap = pl.cm.Reds

        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))

        # Set alpha
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N)

        # Create new colormap
        my_cmap = ListedColormap(my_cmap)   

        # plt.imshow(var_ep.reshape(imsize), cmap=my_cmap)
        # plt.colorbar()
        # plt.show()  

        plt.figure(figsize=(30, 10))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1.06])
        gs.update(wspace=0.02, hspace=0.02)

        ax = plt.subplot(gs[0,0])
        im = ax.imshow(test_image[:, :, 0], cmap='gray')
        im = ax.imshow(gt.reshape(imsize), vmin=0., vmax=3., alpha=0.8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        ax = plt.subplot(gs[0,1])
        im = ax.imshow(test_image[:, :, 0], cmap='gray')
        im = ax.imshow(mean_ep.reshape(imsize), vmin=0., vmax=3., alpha=0.8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        # ax = plt.subplot(gs[0, 2])
        # im = ax.imshow(np.mean(mean_al, axis = -1).reshape(imsize), vmin=0., vmax=3.)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_aspect('equal')
        # ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        ax = plt.subplot(gs[0,2])
        #set_size(11,11)
        im = ax.imshow(test_image[:, :, 0], cmap='gray')
        im2 = ax.imshow(mean_ep.reshape(imsize), vmin=0., vmax=3., alpha=0.75)
        im3 = ax.imshow(var_ep.reshape(imsize), cmap=my_cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax.set_aspect('equal')
        ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        plt.colorbar(im3, cax=cax)
        # ax = plt.subplot(gs[0, 4])
        # im = ax.imshow(np.mean(var_al[:, :, :, 1:], axis = -1).reshape(imsize), cmap=plt.cm.RdBu_r)
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_aspect('equal')
        # ax.tick_params(bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off' )
        # print(volume)
        #plt.show()

        plt.savefig('uncertainty_dense/volume%d.png' %volume)
        plt.clf()
        del mean_ep, var_ep, cmap, gt, test_image
    



