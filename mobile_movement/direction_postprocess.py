import os
import csv
import numpy as np
import re

root_path = '/home/rgu/Documents/MARIO_mobile'
csv_file_path = os.path.join(root_path, 'prediction.csv')

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    prediction = []
    label = []
    loss = []
    for row in csv_reader:
        prediction.append(row[0])
        label.append(row[1])
        loss.append(row[2])


direction = prediction[1::]
all_values = [float(value) for item in direction for value in re.findall(r'[-+]?\d*\.\d+|\d+', item)]
all_values = [0.0 if value == -0. else value for value in all_values]

all_values = np.array(all_values)
all_values = all_values.reshape(-1,5)
all_values = all_values.tolist()
prediction[1::] = all_values

# loss_val = loss[1::]
# all_loss = [round(float(value),2) for item in loss_val for value in re.findall(r'-?\d+\.\d+', item)]
# all_loss = np.array(all_loss)
# all_loss = all_loss.tolist()
# loss[1::] = all_loss


with open(os.path.join(root_path,'direction.csv'), 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for i in range(len(prediction)):
        csv_writer.writerow([prediction[i], label[i], loss[i]])


