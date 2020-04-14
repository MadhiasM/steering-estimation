"""
data import of test_data.csv using pandas
"""

# Google Colab Link: https://colab.research.google.com/drive/1jQXI2rnhFuNpdEHBPr1nVOZr0KLHZooT

import os

import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import plot_model

# Get CSV data from GitHub

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

plot_model(model, to_file='model.png')

pred = model.predict(x)
print(f"Shape: {pred.shape}")
print(pred[0:10])

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y))
print(f"Final score (RMSE): {score}")

# Sample predictions
for i in range(10):
    print(f"{i+1}. Input Data: {x[i]}, Ground Truth Data: {y[i]}, Predicted Output Data: {pred[i]}")