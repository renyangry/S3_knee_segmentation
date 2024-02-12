import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# read data from csv file
root_path = '/home/rgu/Documents/MARIO_mobile'
csv_file_path = os.path.join(root_path, 'large_intervals', 'simplified_45.csv')
csv_file_pat_near = os.path.join(root_path, 'small_intervals', 'simplified_45.csv')
################################################################
#### training/testing = 80:20
# with open(csv_file_path, 'r') as csv_file:
#     root_csv = csv.reader(csv_file)
#     data = []
#     label = []
    
#     for line in root_csv:
#         data.append([float(value) for value in line[0:23]])
#         label.append([float(value) for value in line[23:]])

with open(csv_file_pat_near, 'r') as csv_file:
    root_csv = csv.reader(csv_file)
    data = []
    label = []
    for line in root_csv:
        data.append([float(value) for value in line[0:23]])
        label.append([float(value) for value in line[23:]])       

data = np.array(data)
label = np.array(label)
label_ = np.delete(label,2,axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data_scaled, label_, test_size=0.2, random_state=42)


# training model
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, criterion='squared_error', n_jobs=-1)
# criterion = {squared_error, friedman_mse}

# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = ?, n_iter = 1000, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Save the model
model_filename = os.path.join(root_path, 'random_forest_model.pkl')
joblib.dump(rf, model_filename)

with open(os.path.join(root_path,'prediction.csv'), 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['prediction','label', 'MSE', 'RMSE'])
    csv_file.close()
for i in range(len(X_test)):
    pred = rf.predict(X_test[i].reshape(1,-1))
    predt = [round(x,2) for x in pred.flatten()]
    # gt = [round(x,2) for x in y_test[i]] 
    
    # print(predt)
    # print((y_test[i]))
    # print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test[i].reshape(1,-1), pred))
    # print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test[i].reshape(1,-1), pred))
    # print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test[i].reshape(1,-1), pred)))
    # mape = np.mean(np.abs((y_test[i] - pred) / np.abs(y_test[i].reshape(1,-1))))
    # print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
    # print('Accuracy:', round(100*(1 - mape), 2))
    
    with open(os.path.join(root_path,'prediction.csv'), 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([predt, y_test[i], metrics.mean_squared_error(y_test[i].reshape(1,-1), pred), np.sqrt(metrics.mean_squared_error(y_test[i].reshape(1,-1), pred))])
        csv_file.close()


# sgd_regressor = SGDRegressor()
# svm_regressor = SVR()
# elastic_net = ElasticNet()