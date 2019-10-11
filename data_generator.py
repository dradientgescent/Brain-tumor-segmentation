import numpy as np
import keras
from glob import glob
import random
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, batch_size = 16, channels = 1, seq = None):
        'Initialization'
        self.path = path
        self.batch_size = batch_size
        self.n_channels = channels
        seq_map = {'flair': 0, 't1': 1, 't2': 3, 't1c': 2}
        if seq not in seq_map.keys(): 
            raise ValueError("Unknown seq given, allowed seq values: ['flair', 't1', 't2', 't1c']")
        self.seq = seq_map[seq]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(glob(self.path + '/patches/*.npy')) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Generate data
        #print(indexes_i, indexes_j)

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = []
        y = []

        # Generate data
        while(len(X)) != self.batch_size:
            index_i = random.randint(0, 245)
            index_j = random.randint(0, 438)

            if os.path.exists(self.path + 'patches/patch_%d_%d.npy' %(index_i, index_j)) and os.path.exists(self.path + 'masks/label_%d_%d.npy' %(index_i, index_j)):

                _slice = np.load(self.path + 'patches/patch_%d_%d.npy' %(index_i, index_j))[:, :, self.seq][...,None]

                _slice = _slice/np.max(_slice)
                # Store sample
                X.append(_slice)
                # Store class
                y.append(np.load(self.path + 'masks/label_%d_%d.npy' %(index_i, index_j)))

        #print(np.array(y).shape, np.array(X).shape)
        pad_val = (0, 0)
        pad = ((0, 0), pad_val, pad_val, (0, 0))
       
        return np.pad(np.array(X), pad, mode='constant'), np.pad(np.array(y), pad, mode='constant')

