import coremltools
import joblib
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd



root_path = '/home/rgu/Documents/MARIO_mobile'


csv_file_pat_near = os.path.join(root_path, 'small_intervals', 'simplified_45.csv')
data_pd = pd.read_csv(csv_file_pat_near, header=None)
data = data_pd.iloc[:,0:23].values
label = data_pd.iloc[:, 23:25].join(data_pd.iloc[:, 26:]).values


# with open(csv_file_pat_near, 'r') as csv_file:
#     root_csv = csv.reader(csv_file)
#     # data = []
#     label = []
#     for line in root_csv:
#         data.append([float(value) for value in line[0:23]])
#         label.append([float(value) for value in line[23:]])       
# data = np.array(data)
# label = np.array(label)
# label_ = np.delete(label,2,axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data_scaled, label, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# model = joblib.load(os.path.join(root_path, 'random_forest_model.pkl'))

coreml_model = coremltools.converters.sklearn.convert(rf)

coremltools.models.MLModel(spec).save((os.path.join(root_path, 'RandomForestModel.mlmodel')))