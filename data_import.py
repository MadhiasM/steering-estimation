"""
data import of test_data.csv using pandas
"""

# 1 import raw data using pandas
# 2 convert data into 
# 3
# 4
# 5


import os
import pandas as pd

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


data = df.values

print(data)