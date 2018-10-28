from keras import optimizers, losses, Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dropout, Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import pdb


LOAD_TRAIN = False#True
EPOCH_SIZE = 3

def creat_model(input_shape, class_num):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(5,5,5),activation='relu',input_shape=input_shape)) 
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=1, padding='valid'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2, padding='valid'))
    model.add(Conv3D(128,(5,5,5),activation='relu'))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=1,padding='valid'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2,padding='valid'))
    #model.add(Conv3D(256,(4,4,4),activation='relu'))
    model.add(Conv3D(256,(5,5,5),activation='relu'))
    #model.add(MaxPool3D(pool_size=(2,2,2),strides=1,padding='valid'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2,padding='valid'))
    model.add(Flatten())
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num,activation='softmax'))
    return model


if __name__ == '__main__':
    
    #pdb.set_trace()
    from prepareData import partition
    from prepareData import MATRIX_SIZE
    partition['train']
    partition['validation']

    #from protein_ligand_utils import protein_ligand_reader
    #train_x, train_y, test_x, test_y = protein_ligand_reader()
    
    if LOAD_TRAIN:
        model = load_model("Trainsamples"+str(TOTAL_TRAIN_SAMPLE)+"_Batchsize"+str(BATCH_SIZE)+"_Errsample"+str(ERR_SAMPLE)+"_Matrixsize"+str(MATRIX_SIZE)+"_Samplesperh5"+str(SAMPLES_PERH5)+"_Epoch"+str(EPOCH_SIZE)+"_model.h5")
    else:

        #pdb.set_trace()
        model = creat_model(input_shape=(60,60,60,4),class_num=2)
        #model = creat_model(input_shape=(30,30,30,4),class_num=2)
        
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
        

        from prepareData import TOTAL_TRAIN_SAMPLE, TOTAL_VALIDATION_SAMPLE,BATCH_SIZE,generate_train
        from prepareData import SAMPLES_PERH5, ERR_SAMPLE, MATRIX_SIZE
        model.fit_generator(generator=generate_train(int(BATCH_SIZE)),
                            steps_per_epoch=int(TOTAL_TRAIN_SAMPLE/BATCH_SIZE), 
                            epochs=int(EPOCH_SIZE),
                            use_multiprocessing=True,
                            workers=32)


        #model.fit(x=train_x, y=train_y, epochs=EPOCH_SIZE, verbose=1)
        #pdb.set_trace()
        model.save("Trainsamples"+str(TOTAL_TRAIN_SAMPLE)+"_Batchsize"+str(BATCH_SIZE)+"_Errsample"+str(ERR_SAMPLE)+"_Matrixsize"+str(MATRIX_SIZE)+"_Samplesperh5"+str(SAMPLES_PERH5)+"_Epoch"+str(EPOCH_SIZE)+"_model.h5")
        print("Saved model to disk")

    loss, acc = model.evaluate(x=test_x, y=test_y)
    print('Test accuracy is {:.4f}'.format(acc))
    y_pred = model.predict(test_x) 
    #conf_mat = confusion_matrix(test_y, y_pred)
    #print("Confusion matrix:")
    #print(conf_mat)
