# set some parameters for different validation file
# the location of training dataset
data_path = "../training_data/"

model_file = "v3_Trainsamples20900_Batchsize20_Errsample10_Matrixsize60_Samplesperfile1100_Epoch12_model.h5"

filter_distance_threshold = 75

lig_start_idx = 1
lig_end_idx = 825


# total 824 samples to predict
# set parameters of validation 1
validation_1_pro_start_idx = 1
validation_1_validate_smaples = 200

# set parameters of validation 2
validation_2_pro_start_idx = 201
validation_2_validate_smaples = 200

# set parameters of validation 3
validation_3_pro_start_idx = 401
validation_3_validate_smaples = 200

# set parameters of validation 1
validation_4_pro_start_idx = 601
validation_4_validate_smaples = 224


# the last 800 samples on training dataset
'''
lig_start_idx = 2201
lig_end_idx = 3001

# total 800 samples to predict
# set parameters of validation 1
validation_1_pro_start_idx = 2201
validation_1_validate_smaples = 200

# set parameters of validation 2
validation_2_pro_start_idx = 2401
validation_2_validate_smaples = 200

# set parameters of validation 3
validation_3_pro_start_idx = 2601
validation_3_validate_smaples = 200

# set parameters of validation 1
validation_4_pro_start_idx = 2801
validation_4_validate_smaples = 200
'''