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
import matplotlib.pyplot as plt
import cv2 
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
kernel = np.ones((16, 16), np.uint8) 


# from evaluation_metrics import *

path_HGG = glob('/home/pi/Projects/beyondsegmentation/HGG/**')
path_LGG = glob('/home/pi/Projects/beyondsegmentation/LGG**')

test_path=path_HGG+path_LGG
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
        test_image = np.pad(test_image, pad_width=npad, mode='constant', constant_values=0)
        npad = ((8, 8), (8, 8))
        gt = np.pad(gt, pad_width=npad, mode='constant', constant_values=0)
    return test_image, gt





class Dissector():

    def __init__(self, model, weights, data_path, layer_name):
        model = load_model(model, custom_objects={'gen_dice_loss':gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
        model.load_weights(weights)

        self.model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        for i in range(0, len(self.model.layers)):
            self.model.layers[i].set_weights(model.layers[i].get_weights())
            self.model.layers[i].trainable = False

        self.path = glob(data_path)[:5]
        self.layer_name = layer_name

        pass

    def get_threshold_maps(self, percentile):

        fmaps = []

        for i in range(len(self.path)):
            print (self.path[i])
            input_, label_ = load_vol(self.path[i], 'unet', slice_=78)
            fmaps.append(np.squeeze(self.model.predict(input_[None, ...])))

        fmaps = np.concatenate(fmaps, axis=0)
        # threshold_maps = np.percentile(fmaps, percentile, axis=0)
        threshold_maps = np.mean(fmaps, axis=0)

        np.save('ModelDissection_layer_{}.npy'.format(self.layer_name), threshold_maps)
        return threshold_maps


    def apply_threshold(self, test_image, threshold_maps):

        fmaps = np.squeeze(self.model.predict(test_image[None, ...]))
        masks = fmaps > threshold_maps
        masks = 1.*(masks)

        print (fmaps.max(), fmaps.min(), threshold_maps.min(), threshold_maps.max())

        shape = test_image.shape[:-1]
        resized_masks = np.zeros((shape[0], shape[1], masks.shape[2]))

        for i in range(masks.shape[-1]):
            resized_masks[:,:,i] = (cv2.erode(imresize(masks[:,:,i], shape, interp='nearest'), kernel, iterations=1))/255

        for i in range(5):
            for j in range(5):
                plt.subplot(5, 5, i*5 +(j+1))
                plt.imshow(resized_masks[:,:,i*5 +(j+1)], cmap='gray')


        plt.subplot(5,5,1)
        plt.imshow(test_image[:,:,3])
        plt.subplot(5,5,2)
        plt.imshow(test_image[:,:,3]*np.mean(resized_masks, axis=2), cmap='gray')

        plt.show()

if __name__ == "__main__":

    D = Dissector('/home/pi/Projects/beyondsegmentation/Brain-tumor-segmentation/trained_models/U_resnet/ResUnet.h5', 
                    '/home/pi/Projects/beyondsegmentation/Brain-tumor-segmentation/trained_models/U_resnet/ResUnet.40_0.559.hdf5', 
                    "/home/pi/Projects/beyondsegmentation/LGG/**",
                    "conv2d_17")

    threshold_maps = D.get_threshold_maps(0.995)
    path = glob("/home/pi/Projects/beyondsegmentation/HGG/**")[:5]
    input_, label_ = load_vol(path[3], 'unet', slice_=78)
    input_ = (input_ - np.min(input_))/(np.max(input_) - np.min(input_))
    D.apply_threshold(input_, threshold_maps)
