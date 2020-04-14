"""
data import of test_data.csv using pandas
"""

# 1 Import raw Data using Pandas
# 2 Convert Pandas Tabular Data into Numpy Arrays
# 3
# 4
# 5


import os

import numpy
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

print("The Current directory is " + os.getcwd())
os.chdir("./data")
print("The Current directory is " + os.getcwd())

df = pd.read_csv('test_data.csv')

# print(df[0:5])

# Strip non-numerics
# print(df.dtypes)
df = df.select_dtypes(include=['float64', 'int64'])

# calculate statistical data
# headers = list(df.columns.values)
# fields = []

# for field in headers:
#     fields.append({
#         'name' : field,
#         'mean' : df[field].mean(),
#         'var' : df[field].var(),
#         'std' : df[field].std()
#     })

# for field in fields:
#     print(fields)

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