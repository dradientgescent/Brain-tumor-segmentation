from keras.models import load_model
from glob import glob
import keras
import numpy as np
from losses import *
import random
from extract_patches import Pipeline
from keras.utils import np_utils

class Evaluator():

    def __init__(self):
        pass

    def Evaluate(self, model, weights, data_path):

        model = load_model(model, custom_objects={'gen_dice_loss':gen_dice_loss,'dice_whole_metric':dice_whole_metric,'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})

        model.load_weights(weights)

        path = glob(data_path)


        dice_whole, dice_core, dice_en = 0, 0, 0

        for i in range(len(path)):

            pipe = Pipeline(list_train=path[i:i+1], Normalize=True)

            #print(pipe.train_im.shape)

            X = pipe.train_im[:, :4, :, :, :].reshape((4, 155, 240, 240))

            X = np.transpose(X, (1, 2, 3, 0)).astype(np.float32)

            Y_labels = pipe.train_im[:, 4, :, :, :]

            Y_labels[Y_labels == 4] = 3

            # transform y to one_hot enconding for keras
            shp = Y_labels.shape[0]
            Y_labels = Y_labels.reshape(-1)
            Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
            Y_labels = Y_labels.reshape(155, 240, 240, 4)

            #X = np.pad(X, ((0,0), (8,8), (8,8), (0,0)), 'constant')
            #Y_labels = np.pad(Y_labels, ((0,0), (8,8), (8,8), (0,0)), 'constant')

            dice_instance_whole = 0

            prediction = []
            imshape = (240, 240)


            for j in range(X.shape[0]):

                prediction.append(model.predict(X[j].reshape((1, *(imshape), 4))))

            #print(np.array(prediction).reshape((155, 240, 240, 4)).shape)

            prediction = np.array(prediction).reshape((1, 155, *(imshape), 4))
            Y_labels = Y_labels.reshape((1, 155, *(imshape), 4))

            #(dice_whole_slice, dice_core_slice, dice_en_slice) =  dice_whole_metric(prediction, Y_labels.astype('float64')), dice_core_metric(prediction, Y_labels.astype('float64')), \
            #                                                      dice_en_metric(prediction, Y_labels.astype('float64'))
            dice_whole_slice = soft_dice_loss(prediction[:, :, :, :, 1:], Y_labels[:, :, :, :, 1:])
            dice_core_slice = soft_dice_loss(np.stack((prediction[:, :, :, :, 1], prediction[:, :, :, :, 3]), axis = -1),
                                             np.stack((Y_labels[:, :, :, :, 1], Y_labels[:, :, :, :, 3]), axis = -1))
            dice_en_slice = soft_dice_loss(prediction[:, :, :, :, -1].reshape((1, 155, *(imshape), 1)), Y_labels[:, :, :, :, -1].reshape((1, 155, *(imshape), 1)))

            del prediction
            del X, Y_labels, pipe

            print(dice_whole_slice, dice_core_slice, dice_en_slice)

            #print('1')
            dice_whole += dice_whole_slice
            dice_core += dice_core_slice
            dice_en += dice_en_slice

            #print('Instance Dice Whole Score = ', keras.backend.get_value(dice_whole_slice))

        return(dice_whole/len(path), dice_core/len(path), dice_en/len(path))


if __name__ == "__main__":

    E = Evaluator()

    whole, core, en = E.Evaluate('/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_cc/Unet_without_skip.h5', '/home/parth/Interpretable_ML/Brain-tumor-segmentation/checkpoints/Unet_cc/SimUnet.40_0.060.hdf5', "/home/parth/Interpretable_ML/BraTS_2018/val/**")


    print(whole, core, en)