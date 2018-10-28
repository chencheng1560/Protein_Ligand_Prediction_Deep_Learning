import numpy as np
import sys

ground_truth_filename = 'test_ground_truth_example.txt'
predictions_filename = 'test_predictions_example.txt'
base_idx = 2200
ground_truth_arr = np.loadtxt(ground_truth_filename, dtype=np.int, delimiter='\t', skiprows=1)
predictions_arr = np.loadtxt(predictions_filename, dtype=np.int, delimiter='\t', skiprows=1)

num_samples = ground_truth_arr.shape[0]
num_predictions = predictions_arr.shape[0]

print(num_predictions)
print(num_samples)


if num_predictions > num_samples:
    print('WARNING: number of predictions are higher than number of samples!!!')
    #num_predictions = num_samples
    num_samples = num_predictions

# prepare ground truth dictionary
ground_truth_dict = dict()
for i in range(num_predictions):
    dict_key = base_idx + i#ground_truth_arr[i,0]
    dict_value = base_idx + i#ground_truth_arr[i,1]
    ground_truth_dict[dict_key] = dict_value

print(list(ground_truth_dict.keys())[1])
# count correct predictions
num_correct_pred = 0

for i in range(num_predictions):
    pro_id = predictions_arr[i,0]
    lig_list = list(predictions_arr[i,1:])
    print(pro_id)
    truth_lig_id = ground_truth_dict[pro_id]
    if truth_lig_id in lig_list:
        num_correct_pred += 1

acc = num_correct_pred / num_samples

print('accuracy:{:.3f}'.format(acc))


