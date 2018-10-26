# This module used to validate mode by using a part of training dataset
import numpy as np
from filter import get_distance_complex

if __name__ == '__main__':
    lig_suffix = '_lig_cg.pdb'
    pro_suffix = '_pro_cg.pdb'
    data_path = "../training_data/"
    top10 = 10
    
    validate_smaples = 100
    filter_distance_threshold = 120;


    predictions_top10 = np.zeros([validate_smaples+1, 10+1]).astype(int)


    #iterate every protein
    for i in range(1, validate_smaples+1):
        pro_filename = '{:04}'.format(i) + pro_suffix
        acc_all_lig = []
        print(i)

        # iterate every ligand for each protein
        for k in range(1, validate_smaples + 1):
            lig_filename = '{:04}'.format(k) + lig_suffix
            dist = get_distance_complex(data_path + lig_filename, data_path + pro_filename)
            if(dist < filter_distance_threshold):
                accuarcy = k #accuarcy = predict ()
                acc_all_lig.append(accuarcy)

        sort_acc_idx = np.argsort(acc_all_lig)
        sort_acc_idx = sort_acc_idx[::-1]
        print(sort_acc_idx)

        # only collect top 10 lig idx
        predictions_top10[i][0] = i
        for j in range(0, min(10, len(sort_acc_idx))):
            predictions_top10[i][j+1] = sort_acc_idx[j] + 1

    print(predictions_top10)
    np.savetxt('test.out', predictions_top10, fmt='%i', delimiter='\t')

