import os
import csv
import numpy as np

root_path =  '/home/rgu/Documents/MARIO_mobile'
csv_file_path = os.path.join(root_path, 'direction.csv')

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    prediction = []
    label = []
    loss = []
    for row in csv_reader:
        if row == 0:
            continue
        prediction.append(row[0])
        label.append(row[1])
        loss.append(row[2])


error_count = [float(value) for value in loss[1::] if float(value) > 0.1 and float(value) < 0.9]
print(len(error_count))
print(len(loss))
print(f'the accuracy is: {(len(loss)-len(error_count))/len(loss)*100}')