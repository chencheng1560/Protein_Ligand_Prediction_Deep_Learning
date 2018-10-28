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
    x_train = np.array(X)[train_indices]
    x_train = x_train.transpose([0, 2, 3, 4, 1])
    y_train = keras.utils.to_categorical(np.array(Y)[train_indices],2)
    
    x_test = np.array(X)[test_indices]
    x_test = x_test.transpose([0, 2, 3, 4, 1])
    y_test = keras.utils.to_categorical(np.array(Y)[test_indices],2)

    return x_train, y_train, x_test, y_test 





