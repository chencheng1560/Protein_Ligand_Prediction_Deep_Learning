import pdb
import numpy as np
import random
import h5py
import keras


DATA_PATH = "../training_data/"
H5_PATH = "/data/share/jzhubo/"



MAX_SAMPLE_NUM = 2000
MATRIX_SIZE = 60
ERR_SAMPLE = 10

TOTAL_TRAIN_H5 = 19
TOTAL_VALIDATION_H5 = 1

TOTAL_H5 = TOTAL_TRAIN_H5+TOTAL_VALIDATION_H5
SAMPLES_PERH5 = int((ERR_SAMPLE+1)*MAX_SAMPLE_NUM/TOTAL_H5)
SAMPLES_PERH5_CORRECT = MAX_SAMPLE_NUM/TOTAL_H5

BATCHS_PER_H5 = 55
BATCH_SIZE = SAMPLES_PERH5/BATCHS_PER_H5

if SAMPLES_PERH5 % BATCHS_PER_H5 != 0:
    print("ERR BATCH_SIZE is not align with SAMPLES_PERH5")
    exit()

TOTAL_TRAIN_SAMPLE = TOTAL_TRAIN_H5*SAMPLES_PERH5
TOTAL_VALIDATION_SAMPLE = TOTAL_VALIDATION_H5*SAMPLES_PERH5




partition = {'train': [], 'validation': []}
#partition = {'train': ['0', '1', '2'], 'validation': ['3']}
#pdb.set_trace()
for i in range(0, TOTAL_TRAIN_H5):
    partition['train'].append(i)

for i in range(0, TOTAL_VALIDATION_H5):
    partition['validation'].append(i+TOTAL_TRAIN_H5)

#count=0
#def generate_train():
def generate_train(batch_size):
    #global count
    count=0
    list_x=[]
    list_y=[]
    
    #pdb.set_trace()
    
    while True:
        h5_id = int(count/int(SAMPLES_PERH5))
        hf = h5py.File(H5_PATH+'samples_'+str(h5_id)+'.h5', 'r')
        all_indices = np.arange(int(SAMPLES_PERH5))
        random.seed(h5_id)
        random.shuffle(all_indices)

        current_count=int(count%int(SAMPLES_PERH5))
        #if current_count != count:
            #pdb.set_trace()

        #batch_num = current_count/batch_size
        indices=all_indices[current_count:current_count+int(BATCH_SIZE)]

        for i in range(len(indices)):
            list_x.append(hf['data'][indices[i],])
            list_y.append(hf['label'][indices[i],])

        x= np.array(list_x)
        y = keras.utils.to_categorical(np.array(list_y),2)
        count+=int(BATCH_SIZE)
        list_x=[]
        list_y=[]

        #yield (list_x,y)
        #yield ({'conv3d_1_input':x},{'output':y})
        yield (x,y)
        if count >= TOTAL_TRAIN_SAMPLE:
            count = 0
        
#class DataGenerator(keras.utils.Sequence):
#    'Generates data for Keras'
#    #def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
#                 n_classes=10, shuffle=True):
#        'Initialization'
#        pdb.set_trace()
#        self.dim = dim
#        self.batch_size = batch_size
#        #self.labels = labels
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
#            #X[i,] = np.load('data/' + ID + '.npy')
#
#            # Store class
#            #y[i] = self.labels[ID]
#
#        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
#


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


def prepare_one_sample(lig, pro, size):
    print("[INFO] Prepare the training sample of" + lig + ". pro is " + pro)

    X_c, Y_c, Z_c = get_centre(lig)
    # 30 is big.....
    X_0 = X_c - size/2
    Y_0 = Y_c - size/2
    Z_0 = Z_c - size/2

    PX_list, PY_list, PZ_list, Patomtype_list = read_pdb(pro)
    LX_list, LY_list, LZ_list, Latomtype_list = read_pdb(lig)
    #print(len(PX_list))
    #print(len(LX_list))

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
    #print(sample.shape)
    return sample


def create_training_samples(data_path, h5_path, samples, factor, size, starting_num, h5_id):

    hf = h5py.File(h5_path+'samples_'+str(h5_id)+'.h5', 'w')

    lig_suffix = '_lig_cg.pdb'
    pro_suffix = '_pro_cg.pdb'

    data = []
    label = [] # true is 1, false is 0
    ligand = []
    protein = []
    
    for i in range(int(starting_num+1), int(starting_num + samples +1)):
    #for i in range(1, samples):
        #pdb.set_trace()
        lig_idx = i
        lig_idx='{:04}'.format(lig_idx)

        # prepare a true complex
        pro_filename = lig_idx + pro_suffix
        lig_filename = lig_idx + lig_suffix
        #print('True ' + lig_filename)
        #print(pro_filename)
        true_complex = prepare_one_sample(data_path + lig_filename, data_path + pro_filename, size)
        #pdb.set_trace()
        
        true_complex = true_complex.transpose([1, 2, 3, 0])

        #g = hf.create_group(lig_filename)
        #g.create_dataset('data', data=true_complex)
        #g.create_dataset('label', data='true')
        data.append(true_complex)
        label.append(1)
        print("true pair label is 1")
        ligand.append(int(lig_idx))
        protein.append(int(lig_idx))
        
        # then prepare N = 7 incorrect complex
        for k in range(factor):
            incorrect_pro_idx = random.choice(list(range(1, i-1)) + list(range(i+1, 2000)))
            incorrect_pro_idx = '{:04}'.format(incorrect_pro_idx)
            incorrect_pro_filename = incorrect_pro_idx + pro_suffix
            #print('false ' + lig_filename)
            #print(incorrect_pro_filename)
            false_complex = prepare_one_sample(data_path + lig_filename, data_path + incorrect_pro_filename, size)
            false_complex = false_complex.transpose([1, 2, 3, 0])

            #g = hf.create_group(lig_filename + str(k))
            #g.create_dataset('data', data=false_complex)
            #g.create_dataset('label', data='true')
            
            data.append(false_complex)
            label.append(0)
            print("false pair label is 0")
            ligand.append(int(lig_idx))
            protein.append(int(incorrect_pro_idx))

    hf.create_dataset("data", data =np.array(data))  
    hf.create_dataset("label", data =np.array(label))  
    hf.create_dataset("ligand", data =np.array(ligand))
    hf.create_dataset("protein", data =np.array(protein))
    hf.close()



if __name__ == '__main__':

    #hf = h5py.File('./training_samples.h5', 'w')
    #hf = h5py.File('/data/share/jzhubo/training_samples.h5', 'w')
    
    ##pdb.set_trace()
    for h5_id in range(0, TOTAL_H5):
        starting_num = h5_id * SAMPLES_PERH5_CORRECT
        create_training_samples(DATA_PATH, H5_PATH, SAMPLES_PERH5_CORRECT, ERR_SAMPLE, MATRIX_SIZE, starting_num, h5_id)
    
    #generate evaluation data with different distribution
    ###starting_num = 2850
    ###create_training_samples(DATA_PATH, H5_PATH, 10, 20, MATRIX_SIZE, starting_num, 888)

    #hf.close()

