"""
data import of test_data.csv using pandas
"""
import os


import numpy
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# print("The Current directory is " + os.getcwd())
# os.chdir("./data")
# print("The Current directory is " + os.getcwd())

url = 'https://raw.githubusercontent.com/MadhiasM/steering-estimation/master/data/test_data.csv'
df = pd.read_csv(url)

# Pandas to Numpy

x = df.drop('oversteer', axis=1).values # Input data

y = df['oversteer'].values  # Output data

# Build the Neural Network

model = Sequential()
model.add(Dense(25, input_dim=x.shape[1], activation='relu'))  # Hidden 1
model.add(Dense(10, activation='relu'))                        # Hidden 2
model.add(Dense(1))                                            # Output
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x,y,verbose=2,epochs=100)

# 