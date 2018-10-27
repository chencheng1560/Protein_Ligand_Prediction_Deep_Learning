import h5py as h5
import numpy as np
import math
from sklearn.model_selection import train_test_split
import keras


h5_path = '/data/share/jzhubo/training_samples.h5'

def protein_ligand_reader(test_size =0.2, size=-1):
    all_h5 = h5.File(h5_path, "r")
    X = all_h5["data"]
    Y = np.array(all_h5["label"])

    flag = []
    for each in Y:
        if math.isnan(each):
            flag.append(False)
        else:
            flag.append(True)
    flag = np.array(flag)
    
    if size == -1:
        size = X.shape[0]

    #add by cc
    #import pdb; pdb.set_trace()

    # discard 'nan' rows
    all_indices = np.arange(size)[flag]
    
    # random_state = 42, which determine the split result
    train_indices, test_indices = train_test_split(all_indices, test_size=test_size,random_state=42)
    #return X, Y, train_indices, test_indices
    x_train = np.array(X)[train_indices]
    x_train = x_train.transpose([0, 2, 3, 4, 1])
    y_train = keras.utils.to_categorical(np.array(Y)[train_indices],2)
    
    x_test = np.array(X)[test_indices]
    x_test = x_test.transpose([0, 2, 3, 4, 1])
    y_test = keras.utils.to_categorical(np.array(Y)[test_indices],2)

    return x_train, y_train, x_test, y_test 



#import numpy as np
#import keras
#
#class DataGenerator(keras.utils.Sequence):
#    'Generates data for Keras'
#    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#                 n_classes=10, shuffle=True):
#        'Initialization'
#        self.dim = dim
#        self.batch_size = batch_size
#        self.labels = labels
#        self.list_IDs = list_IDs
#        self.n_channels = n_channels
#        self.n_classes = n_classes
#        self.shuffle = shuffle
#        self.on_epoch_end()
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        # Generate indexes of the batch
#        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#        # Find list of IDs
#        list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#        # Generate data
#        X, y = self.__data_generation(list_IDs_temp)
#
#        return X, y
#
#    def on_epoch_end(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.list_IDs))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)
#
#    def __data_generation(self, list_IDs_temp):
#        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#        # Initialization
#        X = np.empty((self.batch_size, *self.dim, self.n_channels))
#        y = np.empty((self.batch_size), dtype=int)
#
#        # Generate data
#        for i, ID in enumerate(list_IDs_temp):
#            # Store sample
#            X[i,] = np.load('data/' + ID + '.npy')
#
#            # Store class
#            y[i] = self.labels[ID]
#
#        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
