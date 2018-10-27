# filter out some distant samples
# input is a lig and pro
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import h5py
from prepareData import read_pdb

def get_centre_filter(filename):
    X_list, Y_list, Z_list, atomtype_list=read_pdb(filename)
    #print(X_list, Y_list, Z_list, atomtype_list)
    #X_c = np.mean(np.array(X_list))
    #Y_c = np.mean(np.array(Y_list))
    #Z_c = np.mean(np.array(Z_list))
    #X_max = np.amax(np.array(X_list))
    # may change to mean to see the performance
    X_c = (np.amin(np.array(X_list)) + np.amax(np.array(X_list)))/2
    Y_c = (np.amin(np.array(Y_list)) + np.amax(np.array(Y_list)))/2
    Z_c = (np.amin(np.array(Z_list)) + np.amax(np.array(Z_list)))/2
    #print(X_c)
    #print(Z_c)
    return X_c, Y_c, Z_c


def get_distance_complex(lig, pro):
    #print("[INFO] checking the distance of " + lig + pro + " complex.")
    X_c_l, Y_c_l, Z_c_l = get_centre_filter(lig)
    X_c_p, Y_c_p, Z_c_p = get_centre_filter(pro)
    dist = math.sqrt((X_c_l - X_c_p) ** 2 + (Y_c_l - Y_c_p) ** 2 + (Z_c_l - Z_c_p) ** 2)
    return dist


if __name__ == '__main__':
    lig_suffix = '_lig_cg.pdb'
    pro_suffix = '_pro_cg.pdb'
    data_path = "../training_data/"



    dist_array = []
    for i in range(1, 3000):
        lig_idx = i
        lig_idx='{:04}'.format(lig_idx)
        lig_filename = lig_idx + lig_suffix

        incorrect_pro_idx = random.choice(list(range(1, i - 1)) + list(range(i + 1, 3000)))
        incorrect_pro_idx = '{:04}'.format(incorrect_pro_idx)
        pro_filename = incorrect_pro_idx + pro_suffix

        dist = get_distance_complex(data_path+lig_filename, data_path+pro_filename)
        dist_array.append(dist)


    fig, ax = plt.subplots()
    ax.set(xlabel='complex', ylabel='distance',
           title='The distance of different false complex')
    ax.plot(dist_array)
    plt.show()











