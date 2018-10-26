from keras import optimizers, losses
from keras.layers import Input, Conv3D, MaxPool3D, Flatten, Dropout, Dense
from PIL import Image
import matplotlib.pyplot as plt

def creat_model(input_shape, class_num):
    model = Sequential()
    model.add(Conv3D(64, kernel_size=(5,5,5),activation='relu',input_shape=input_shape))
    
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2, padding='valid'))
    model.add(Conv3D(128,(5,5,5),activation='relu'))
    model.add(MaxPool3D(pool_size=(2,2,2),strides=2,padding='valid'))
    model.add(Conv3D(256,(5,5,5),activation='relu'))
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

    from protein_ligand_utils import protein_ligand_reader
    train_x, train_y, test_x, test_y = protein_ligand_reader()
    md = creat_model(input_shape=(60,60,60,4),class_num=2)
    sg = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    md.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])
    #md.compile(optimizer=sgd, loss='mean_squared_error')
    model.fit(x=train_x, y=train_y, epochs=50, verbose=1)
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print('Test accuracy is {:.4f}'.format(acc))
