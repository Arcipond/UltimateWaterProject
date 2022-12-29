from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

data = pd.read_csv("dataset.csv")

replace_dict = {'Color':{'Colorless':1, 'Near Colorless':2, 'Faint Yellow':3, 'Light Yellow':4, 'Yellow':5}, \
    'Source':{'Aquifer':1, 'River':2, 'Lake':3, 'Reservoir':4, 'Ground':5, 'Stream':6, 'Well':7, 'Spring':8}, \
    'Month': {"January":1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12},
}

data = data.replace(replace_dict)

for column in data.columns:
    data[column] = data[column].fillna(data[column].mean())


data.replace()

x=data.iloc[:,2:-1]            
y=data['Target']

x_train_all, x_test, y_train_all, y_test = \
  train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)
x_train, x_val, y_train, y_val = \
  train_test_split(x_train_all,y_train_all,stratify=y_train_all, \
                   test_size=0.2,random_state=42)


scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)  # standard preprocessing
x_val_scaled = scaler.transform(x_val)

mlp = MLPClassifier(hidden_layer_sizes=(10,10), activation='logistic', \
                    solver='adam', alpha=0.005, batch_size=32, \
                    learning_rate_init=0.1, max_iter=500)

mlp.fit(x_train_scaled, y_train)

print(f"CLassifier Score : {mlp.score(x_val_scaled, y_val)}")
