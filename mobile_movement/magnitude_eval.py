import os
import csv
import numpy as np
import re


def sign_check(all_pred, all_label):
    sign = []
    for i in range(len(all_pred)):
        if all_pred[i] > 0 and all_label[i] > 0:
            sign.append(True)
        elif all_pred[i] < 0 and all_label[i] < 0:
            sign.append(True)
        # elif all_pred[i] == 0 and all_label[i] == 0:
        #     sign.append(True)
        elif all_pred[i] == 0:
            sign.append(True)
        elif all_label[i] == 0:
            sign.append(True)
        else:
            sign.append(False)
    return sign

# loading data 
root_path = '/home/rgu/Documents/MARIO_mobile'
csv_file_path = os.path.join(root_path, 'prediction.csv')

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    prediction = []
    label = []
    # loss = []
    for row in csv_reader:
        prediction.append(row[0])
        label.append(row[1])
        # loss.append(row[2])

# data cleaning and processing
magnitude = prediction[1::]
all_pred = [float(value) for item in magnitude for value in re.findall(r'[-+]?\d*\.\d+|\d+', item)]
all_pred = [0.0 if value == -0. else value for value in all_pred]

magnitude_r = label[1::]
all_label = [float(value) for sublist in magnitude_r for value in re.findall(r'[-+]?\d+\.', sublist)]
# calculate magnitude difference
magnitude_diff = [round((all_pred[num_mag]-all_label[num_mag]),2) for num_mag in range(len(all_pred))]
magnitude_diff = np.array(magnitude_diff)
magnitude_diff = magnitude_diff.reshape(-1,5)
magnitude_diff = magnitude_diff.tolist()

threshold = 1.0 # the maxmimal mm allowed, can be changed 
mag_diff = []
mag_diff_idx = []
idx_zero = 0
idx_one = 0
idx_two = 0
idx_three = 0
idx_four = 0
for i in range(len(magnitude_diff)):
    condition = [True if np.abs(value) < threshold else False for value in magnitude_diff[i]]
    cond_count = condition.count(False)
    cond_false = [idx for idx, x in enumerate(condition) if x == False]
    if 1 in cond_false:
        idx_one += 1
    elif 2 in cond_false:
        idx_two += 1
    elif 3 in cond_false:
        idx_three += 1
    elif 4 in cond_false:
        idx_four += 1
    elif 0 in cond_false:
        idx_zero += 1
    mag_diff.append(cond_count)
    mag_diff_idx.append(cond_false)
    
magnitude_diff.insert(0, ['magnitude_diff'])

mag_accu = (mag_diff.count(0)/len(mag_diff)) * 100
print(f"Magnitude Accuracy with in {threshold} mm: {mag_accu} %")
print(f"Magnitude Error for x-axis translation: {idx_zero/len(mag_diff)*100} %")
print(f"Magnitude Error for y-axis translation: {idx_one/len(mag_diff)*100} %")
print(f"Magnitude Error for x-axis rotation: {idx_two/len(mag_diff)*100} %")
print(f"Magnitude Error for y-axis rotation: {idx_three/len(mag_diff)*100} %")
print(f"Magnitude Error for z-axis rotation: {idx_four/len(mag_diff)*100} %")


mag_diff.insert(0, ['no_of_mag_diff'])
mag_diff_idx.insert(0, ['mag_diff_idx'])


# calculate sign difference
sign_diff = sign_check(all_pred, all_label)
sign_diff = np.array(sign_diff)
sign_diff = sign_diff.reshape(-1,5)
sign_diff = sign_diff.tolist()

direction_diff = []
direction_diff_idx = []
zero_count = 0
one_count = 0
two_count = 0
three_count = 0
four_count = 0
#  i want to count how many 0, 1, 2, 3, 4 in direction_diff_idx
for i in range(len(sign_diff)):
    sign_count = sign_diff[i].count(False)
    idx_false = [idx for idx, x in enumerate(sign_diff[i]) if x == False]
    if 1 in idx_false:
        one_count += 1
    elif 2 in idx_false:
        two_count += 1
    elif 3 in idx_false:
        three_count += 1
    elif 4 in idx_false:
        four_count += 1
    elif 0 in idx_false:
        zero_count += 1
    direction_diff.append(sign_count)
    direction_diff_idx.append(idx_false)

sign_diff.insert(0, ['direction_diff'])

direction_accu = (direction_diff.count(0)/len(direction_diff)) * 100
print(f"Direction Accuracy: {direction_accu} %")
print(f"Direction Error for x-axis translation: {zero_count/len(direction_diff)*100} %")
print(f"Direction Error for y-axis translation: {one_count/len(direction_diff)*100} %")
print(f"Direction Error for x-axis rotation: {two_count/len(direction_diff)*100} %")
print(f"Direction Error for y-axis rotation: {three_count/len(direction_diff)*100} %")
print(f"Direction Error for z-axis rotation: {four_count/len(direction_diff)*100} %")
    

direction_diff.insert(0, ['no_of_direction_diff'])
direction_diff_idx.insert(0, ['direction_diff_idx'])


# save all computed difference in csv file
with open(os.path.join(root_path,'mag_results_ignore0.csv'), 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for i in range(len(magnitude_diff)):
        csv_writer.writerow([magnitude_diff[i],mag_diff[i], mag_diff_idx[i], sign_diff[i], direction_diff[i],direction_diff_idx[i]])
