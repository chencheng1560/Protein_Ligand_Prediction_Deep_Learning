# This module used to validate mode by using a part of training dataset
from keras import optimizers, losses, Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dropout, Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
from filter import get_distance_complex
from prepareData import prepare_one_sample

if __name__ == '__main__':
    lig_suffix = '_lig_cg.pdb'
    pro_suffix = '_pro_cg.pdb'
    data_path = "../training_data/"
    # some parameters
    top10 = 10
    base_index = 2200
    validate_smaples = 30
    filter_distance_threshold = 120
    # import the model
    model = load_model("100complements_32epochs_model.h5")
    # the first row is for pred 1... pre 2....
    # the first col is for protein index ....
    predictions_top10 = np.zeros([validate_smaples+1, top10+1]).astype(int)

    num_correct_pred = 0
    row = 0 # record the rows of predictions_top10
    # iterate every protein
    for i in range(base_index, validate_smaples + base_index):
        pro_filename = '{:04}'.format(i) + pro_suffix
        acc_all_lig = []
        print(i)
        # iterate every ligand for each protein
        for k in range(base_index, validate_smaples + base_index):
            lig_filename = '{:04}'.format(k) + lig_suffix
            dist = get_distance_complex(data_path + lig_filename, data_path + pro_filename)

            if(dist < filter_distance_threshold):
                complex = prepare_one_sample(data_path + lig_filename, data_path + pro_filename, 60)
                #complex = complex.transpose([0, 2, 3, 4, 1])
                complex = np.transpose(complex)
                complex = np.expand_dims(complex, 0)
                # predict
                y_pred = model.predict(complex)
                acc = y_pred[0][1]
                acc_all_lig.append(acc)
                print(acc)
            else:
                acc = 0
                acc_all_lig.append(acc)

        sort_acc_idx = np.argsort(acc_all_lig)
        sort_acc_idx = sort_acc_idx[::-1]  #reverse

        # only collect top 10 lig idx
        row += 1
        predictions_top10[row][0] = i
        for j in range(0, min(10, len(sort_acc_idx))):
            predictions_top10[row][j+1] = sort_acc_idx[j] + base_index
            if((sort_acc_idx[j]+base_index) == i):
                num_correct_pred += 1

    acc = num_correct_pred / validate_smaples
    print('accuracy:{:.3f}'.format(acc))

    np.savetxt('test_predictions_example.txt', predictions_top10, fmt='%i', delimiter='\t')

