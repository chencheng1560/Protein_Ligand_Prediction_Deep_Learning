from keras import optimizers, losses, Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dropout, Dense
from keras.models import load_model
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
import keras
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import h5py
import numpy as np


from prepareData import partition,generate_train
from prepareData import TOTAL_TRAIN_SAMPLE, TOTAL_VALIDATION_SAMPLE,BATCH_SIZE
from prepareData import SAMPLES_PERH5, ERR_SAMPLE, MATRIX_SIZE,H5_PATH

LOAD_TRAIN = False#True
MULTIPLE_GPU = False
EPOCH_SIZE = 24

def creat_model(input_shape, class_num):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(8,8,8),activation='relu',input_shape=input_shape)) 
    #model.add(Conv3D(64, kernel_size=(5,5,5),activation='relu',input_shape=input_shape)) 
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=1, padding='valid'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2, padding='valid'))
    model.add(Conv3D(128,(4,4,4),activation='relu'))
    #model.add(Conv3D(128,(5,5,5),activation='relu'))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=1,padding='valid'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2,padding='valid'))
    model.add(Conv3D(256,(2,2,2),activation='relu'))
    #model.add(Conv3D(256,(5,5,5),activation='relu'))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=1,padding='valid'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2,padding='valid'))
    model.add(Flatten())
    model.add(Dense(800,activation='relu'))
    #model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(400,activation='relu'))
    #model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num,activation='softmax'))
    return model

def load_eval_h5(path):
    print("load evaluation file %s",path)
    hf = h5py.File(H5_PATH+path,'r')
    x = np.array(hf['data'])
    y = keras.utils.to_categorical(np.array(hf['label']),2) 
    return x,y
    

if __name__ == '__main__':
    
    #pdb.set_trace()
    partition['train']
    partition['validation']

    #from protein_ligand_utils import protein_ligand_reader
    #train_x, train_y, test_x, test_y = protein_ligand_reader()
    
    if LOAD_TRAIN:
        if MULTIPLE_GPU:
            h5f = "../model/v1_MultiGpu_Trainsamples"+str(TOTAL_TRAIN_SAMPLE)+"_Batchsize"+str(BATCH_SIZE)+"_Errsample"+str(ERR_SAMPLE)+"_Samplesperfile"+str(SAMPLES_PERH5)+"_Epoch"+str(EPOCH_SIZE)+"_model.h5"
        else:
            h5f = "../model/v1_Trainsamples"+str(TOTAL_TRAIN_SAMPLE)+"_Batchsize"+str(BATCH_SIZE)+"_Errsample"+str(ERR_SAMPLE)+"_Matrixsize"+str(MATRIX_SIZE)+"_Samplesperfile"+str(SAMPLES_PERH5)+"_Epoch"+str(EPOCH_SIZE)+"_model.h5"
        
        #pdb.set_trace()
        
        model = load_model(h5f)
    else:

        # multi-gpu training
        if MULTIPLE_GPU:
            #pdb.set_trace()
            #import os;os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
            import os;os.environ["CUDA_VISIBLE_DEVICES"]="8,9,10,11"
            #import os;os.environ["CUDA_VISIBLE_DEVICES"]="12,13,14,15"
            #import os;os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
            #import os;os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
            model = creat_model(input_shape=(60,60,60,4), class_num=2)
            #model = creat_model(input_shape=(30,30,30,4),class_num=2)
            sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        
            model = multi_gpu_model(model, gpus=4)
            model.compile(loss='categorical_crossentropy',
                                   optimizer=sgd,metrics=['accuracy'])
            
            model.fit_generator(generator=generate_train(int(BATCH_SIZE)),
                                steps_per_epoch=int(TOTAL_TRAIN_SAMPLE/BATCH_SIZE), 
                                epochs=int(EPOCH_SIZE),
                                use_multiprocessing=True,
                                workers=4)
            h5f = "../model/v1_MultiGpu_Trainsamples"+str(TOTAL_TRAIN_SAMPLE)+"_Batchsize"+str(BATCH_SIZE)+"_Errsample"+str(ERR_SAMPLE)+"_Samplesperfile"+str(SAMPLES_PERH5)+"_Epoch"+str(EPOCH_SIZE)+"_model.h5"
            model.save(h5f)
        else:
            #single GPU training
            #pdb.set_trace()
            import os;os.environ["CUDA_VISIBLE_DEVICES"]="11"
            model = creat_model(input_shape=(60,60,60,4),class_num=2)
        
            sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        

            model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
            model.fit_generator(generator=generate_train(int(BATCH_SIZE)),
                             steps_per_epoch=int(TOTAL_TRAIN_SAMPLE/BATCH_SIZE), 
                             epochs=int(EPOCH_SIZE),
                             use_multiprocessing=True,
                             workers=32)
            #model.fit(x=train_x, y=train_y, epochs=EPOCH_SIZE, verbose=1)
            #pdb.set_trace()
            h5f = "../model/v1_Trainsamples"+str(TOTAL_TRAIN_SAMPLE)+"_Batchsize"+str(BATCH_SIZE)+"_Errsample"+str(ERR_SAMPLE)+"_Matrixsize"+str(MATRIX_SIZE)+"_Samplesperfile"+str(SAMPLES_PERH5)+"_Epoch"+str(EPOCH_SIZE)+"_model.h5"
            model.save(h5f)
            print("Saved model to disk")

    eval_files=['samples_eval_100sam_10erreachsam.h5']
    #eval_files=['samples_eval_10sam_10erreachsam.h5','samples_eval_10sam_20erreachsam.h5','samples_eval_10sam_50erreachsam.h5','samples_eval_10sam_100erreachsam.h5','samples_eval_100sam_10erreachsam.h5']
    for file_name in eval_files:
        test_x, test_y = load_eval_h5(file_name) 
        loss, acc = model.evaluate(x=test_x, y=test_y)
        print('Test accuracy is {:.4f}'.format(acc))
        y_pred = model.predict(test_x) 
        #conf_mat = confusion_matrix(test_y, y_pred)
        #print("Confusion matrix:")
    #print(conf_mat)
