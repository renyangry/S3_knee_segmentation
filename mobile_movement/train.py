import os
import csv
from tkinter import X, Y
from attr import validate
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from movement_prediction_model import *
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# read data from csv file
root_path = '/home/rgu/Documents/MARIO_mobile'
csv_file_path = os.path.join(root_path, 'small_intervals', 'simplified_45.csv')
csv_file_pat_near = os.path.join(root_path, 'small_intervals', 'simplified_45.csv')
################################################################
#### training/testing = 80:20
with open(csv_file_path, 'r') as csv_file:
    root_csv = csv.reader(csv_file)
    data = []
    label = []
    
    for line in root_csv:
        data.append([float(value) for value in line[0:23]])
        label.append([float(value) for value in line[23:]])
        
with open(csv_file_pat_near, 'r') as csv_file:
    root_csv = csv.reader(csv_file)
    
    for line in root_csv:
        data.append([float(value) for value in line[0:23]])
        label.append([float(value) for value in line[23:]])   
        
data = np.array(data)
label = np.array(label)
label_ = np.delete(label,2,axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, label_, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

################################################################
#### manual split training/testing dataset 
# with open(csv_file_path, 'r') as csv_file:
#     root_csv = csv.reader(csv_file)
#     x_test = []
#     y_test = []
#     x_train = []
#     y_train = []
    
#     for line in root_csv:
#         assess = line[23:]
#         if any(np.abs(float(value)) > 6 for value in assess):
#             x_test.append([float(value) for value in line[0:23]])
#             y_test.append([float(value) for value in line[23:]])
#         else:
#             x_train.append([float(value) for value in line[0:23]])
#             y_train.append([float(value) for value in line[23:]])

# x_test = np.array(x_test) # (484, 23)
# y_test = np.array(y_test) # (484, 6)
# x_train = np.array(x_train) # (727, 23)
# y_train = np.array(y_train) # (727, 6)

# data = np.concatenate((x_train,x_test), axis=0)
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)

# X_train = data_scaled[0:(x_train.shape[0]-7)] # (720, 23)
# Y_train = y_train[:-7] # (720, 6)
# X_test = data_scaled[(x_train.shape[0]-7):] # (491, 23)
# Y_test = np.concatenate((y_train[-7:], y_test), axis=0) # (491, 6)

# Y_train_ = np.delete(Y_train,2,axis=1)
# Y_test_ = np.delete(Y_test,2,axis=1)

# # data loading in pytorch 
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(Y_train_, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(Y_test_, dtype=torch.float32)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# training starts here 
# model = MovPredictFNN(input_size=X_train_tensor.shape[1], hidden_size=64, output_size=y_train_tensor.shape[1])
model = MovPredictCNN1D()
# model = MovPredictMLP(input_size=X_train_tensor.shape[1], hidden_size1=64, hidden_size2=32, output_size=y_train_tensor.shape[1])

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 5000 
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = []
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
    if (epoch + 1) % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        validate_loss = []
        val_loss = 0
        
        with open(os.path.join(root_path,'prediction.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['prediction','label', 'SmoothL1Loss'])
            csv_file.close()
            
        for val_inputs, val_targets in test_loader:
            val_outputs = model(val_inputs)
            val_loss_ = criterion(val_outputs, val_targets)
            val_loss += val_loss_.item()

            with open(os.path.join(root_path,'prediction.csv'), 'a') as csv_file:
                writer = csv.writer(csv_file)
                predt = val_outputs.flatten().detach().cpu().numpy()
                predt = [round(x,2) for x in predt]
                writer.writerow([predt, val_targets.flatten().detach().cpu().numpy(), val_loss_.item()])
                # writer.writerow([val_outputs.flatten().detach().cpu().numpy(), val_targets.flatten().detach().cpu().numpy()])
                csv_file.close()        
                
        val_loss /= len(test_loader)      
        validate_loss.append(val_loss)  
        
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {val_loss}')
    # print(f'the label is {val_targets.detach().cpu().numpy()}')
    # print(f'the prediction is {val_outputs.detach().cpu().numpy()}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(root_path,'best_model.pth'))
        print(f'Best validation loss so far. Saving model at epoch {epoch + 1}')

    # # Check for early stopping based on validation loss
    # if epoch > 50 and val_loss > best_val_loss and val_loss <1.0:
    #     print(f'Early stopping at epoch {epoch + 1} to prevent overfitting.')
    #     break


# torch.save(model.state_dict(), os.path.join(root_path,'final_model.pth'))

# plt.figure()
# plt.plot(train_loss)
# plt.plot(validate_loss)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train_loss','validate_loss'])
# plt.show()



# for future testing time 
# best_model = MovementPredictionNN()
# best_model.load_state_dict(torch.load(os.path.join(root_path,'best_model.pth')))

# best_model.eval()
# with torch.no_grad():
#     test_outputs = best_model(X_test_tensor)
#     loss = criterion(test_outputs, y_test_tensor)
#     print(f'F1 loss on Testing Set: {loss.item()}')
    
