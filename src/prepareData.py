import pdb
import numpy as np
import random
import h5py
def read_pdb(filename):

    with open(filename, 'r') as file:
        strline_L = file.readlines()
        # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))

        X_list.append(float(stripped_line[30:38].strip()))
        Y_list.append(float(stripped_line[38:46].strip()))
        Z_list.append(float(stripped_line[46:54].strip()))

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


# X_list, Y_list, Z_list, atomtype_list=read_pdb("/Users/mustafauo/Dropbox/NUS_Academic/NUS_2018_2019_1/CS5242/project/data/training_data/2060_lig_cg.pdb")
# X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_pro_cg.pdb")
# X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_lig_cg.pdb")
#print(X_list)
#print(Y_list)
#print(Z_list)
#print(atomtype_list)


def get_centre(filename):
    X_list, Y_list, Z_list, atomtype_list=read_pdb(filename)
    print(X_list, Y_list, Z_list, atomtype_list)
    #X_c = np.mean(np.array(X_list))
    #Y_c = np.mean(np.array(Y_list))
    #Z_c = np.mean(np.array(Z_list))
    #X_max = np.amax(np.array(X_list))
    # may change to mean to see the performance
    X_c = (np.amin(np.array(X_list)) + np.amax(np.array(X_list)))/2
    Y_c = (np.amin(np.array(Y_list)) + np.amax(np.array(Y_list)))/2
    Z_c = (np.amin(np.array(Z_list)) + np.amax(np.array(Z_list)))/2
    print(X_c)
    print(Z_c)
    return X_c, Y_c, Z_c


def prepare_one_sample(lig, pro, size):
    print("[INFO] Prepare the training sample of" + lig + ".")

    X_c, Y_c, Z_c = get_centre(lig)
    # 30 is big.....
    X_0 = X_c - size/2
    Y_0 = Y_c - size/2
    Z_0 = Z_c - size/2

    PX_list, PY_list, PZ_list, Patomtype_list = read_pdb(pro)
    LX_list, LY_list, LZ_list, Latomtype_list = read_pdb(lig)
    print(len(PX_list))
    print(len(LX_list))

    image0 = np.zeros([size,size,size])
    image1 = np.zeros([size,size,size])
    for i in range (len(PX_list)-1):
        x=PX_list[i]
        y=PY_list[i]
        z=PZ_list[i]
        t=Patomtype_list[i]
        if((np.absolute(x-X_c) < size/2) & (np.absolute(y-Y_c) < size/2) & (np.absolute(z-Z_c) < size/2)):
            if(t == 'p'):
                image0[min(int(x - X_0), size-1)][min(int(y-Y_0),size-1)][min(int(z-Z_0), size-1)] = 1
            else:
                image1[min(int(x - X_0), size-1)][min(int(y-Y_0),size-1)][min(int(z-Z_0), size-1)] = 1

    image2 = np.zeros([size, size, size])
    image3 = np.zeros([size, size, size])
    for i in range (len(LX_list)-1):

        x=LX_list[i]
        y=LY_list[i]
        z=LZ_list[i]
        t=Latomtype_list[i]

        if ((np.absolute(x - X_c) < size/2) & (np.absolute(y - Y_c) < size/2) & (np.absolute(z - Z_c) < size/2)):
            if (t == 'p'):
                image2[min(int(x - X_0), size-1)][min(int(y-Y_0),size-1)][min(int(z-Z_0), size-1)] = 1
            else:
                image3[min(int(x - X_0), size-1)][min(int(y-Y_0),size-1)][min(int(z-Z_0), size-1)] = 1

    #print(image2)
    #print(image3)
    sample = np.array((image0, image1, image2, image3))
    print(sample.shape)
    return sample


def create_training_samples(data_path, hf, samples, factor, size):
    training_complex_number = 2000
    validation_complex_number = 1000

    lig_suffix = '_lig_cg.pdb'
    pro_suffix = '_pro_cg.pdb'

    for i in range(1, samples):
        lig_idx = i
        lig_idx='{:04}'.format(lig_idx)

        # prepare a true complex
        pro_filename = lig_idx + pro_suffix
        lig_filename = lig_idx + lig_suffix
        print('True ' + lig_filename)
        print(pro_filename)
        true_complex = prepare_one_sample(data_path + lig_filename, data_path + pro_filename, size)

        g = hf.create_group(lig_filename)
        g.create_dataset('data', data=true_complex)
        g.create_dataset('label', data='true')

        # then prepare N = 7 incorrect complex
        for k in range(factor):
            incorrect_pro_idx = random.choice(list(range(1, i-1)) + list(range(i+1, 2000)))
            incorrect_pro_idx = '{:04}'.format(incorrect_pro_idx)
            incorrect_pro_filename = incorrect_pro_idx + pro_suffix
            print('false ' + lig_filename)
            print(incorrect_pro_filename)
            false_complex = prepare_one_sample(data_path + lig_filename, data_path + incorrect_pro_filename, size)

            g = hf.create_group(lig_filename + str(k))
            g.create_dataset('data', data=false_complex)
            g.create_dataset('label', data='true')


if __name__ == '__main__':

    data_path = "../training_data/"
    samples = 20
    size = 30
    factor = 2

    hf = h5py.File('./training_samples.h5', 'w')
    create_training_samples(data_path, hf, samples, factor, size)
    hf.close()