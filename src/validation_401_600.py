# This module used to validate mode by using a part of training dataset
from keras import optimizers, losses, Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dropout, Dense
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
from filter import get_distance_complex
from prepareData import prepare_one_sample
import configuration

data_path = configuration.data_path

if __name__ == '__main__':
    lig_suffix = '_lig_cg.pdb'
    pro_suffix = '_pro_cg.pdb'
    # some parameters
    top10 = 10
    base_index = configuration.validation_3_pro_start_idx
    validate_smaples = configuration.validation_3_validate_smaples
    filter_distance_threshold = configuration.filter_distance_threshold
    # import the model
    model_file = configuration.model_file
    model = load_model(model_file)
    # the first row is for pred 1... pre 2....
    # the first col is for protein index ....
    predictions_top10 = np.zeros([validate_smaples + 1, top10 + 1]).astype(int)
    num_correct_pred = 0
    row = 0  # record the rows of predictions_top10
    # iterate every protein
    for i in range(base_index, validate_smaples + base_index):
        pro_filename = '{:04}'.format(i) + pro_suffix
        complexes = []
        record_lig_idx = []
        print(i)
        # iterate every ligand for each protein
        for k in range(configuration.lig_start_idx, configuration.lig_end_idx):
            lig_filename = '{:04}'.format(k) + lig_suffix
            dist = get_distance_complex(data_path + lig_filename, data_path + pro_filename)
            if (dist < filter_distance_threshold):
                record_lig_idx.append(k)
                new_complex = prepare_one_sample(data_path + lig_filename, data_path + pro_filename, 60)
                # new_complex = np.transpose(new_complex)
                new_complex = new_complex.transpose([1, 2, 3, 0])
                complexes.append(new_complex)

        # predict
        complexes = np.array(complexes)
        y_pred = model.predict(complexes)
        acc_all_lig = y_pred[:, 1]
        sort_acc_idx = np.argsort(acc_all_lig)
        sort_acc_idx = sort_acc_idx[::-1]  # reversed

        # only collect top 10 lig idx
        row += 1
        predictions_top10[row][0] = i
        for j in range(0, min(10, len(sort_acc_idx))):
            predictions_top10[row][j + 1] = record_lig_idx[sort_acc_idx[j]]
            if (record_lig_idx[sort_acc_idx[j]] == i):
                num_correct_pred += 1
                print('current correct number', num_correct_pred)

    # calculate final acc
    acc = num_correct_pred / validate_smaples
    print('accuracy:{:.3f}'.format(acc))

    np.savetxt('v3_batch_predictions_on_' + model_file + '.txt', predictions_top10, fmt='%i', delimiter='\t')

