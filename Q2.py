import torch

'''
read in the set of values from 0 to 2pi for each x as a 
dataset and train the network to predict the output without seeing the function.
So test using other values to see if over fit,
maybe plot output?

'''
# approximate y = sin(2x1 + 2.0)cos(o.5x2) + 0.5
# x1, x2 -> [0, 2pi]

# Fully connected
# 6 hidden neurons
# threshold connection for all hidden nodes and output node
# Sigmoid activation for hidden
# Linear activation output

# 2.2
import random
import pandas as pd
from math import sin, cos, pi
import numpy as np
import os
import matplotlib.pyplot as plt

def f(x1, x2):
    return sin(2*(x1) + 2.0) * cos(0.5 * x2) + 0.5

df = pd.DataFrame(columns=['x1', 'x2', 'output'])


for i in range(21):
    x1 = random.uniform(0, 2*pi)
    x2 = random.uniform(0, 2*pi)

    y = f(x1, x2)
    df.loc[i] = [x1, x2, y]

# print(df.head())
train = df[:11]
test = df[11:]
# print(train)
# print(test)

if not os.path.isfile('./train.dat') and not os.path.isfile('./test.dat'):
    print('Saving dat files.')
    train.to_csv('./train.dat', index=False, sep=' ', header=False)
    test.to_csv('./test.dat', index=False, sep=' ', header=False)

# train.sort_values(by='x1').plot()
# test.sort_values(by='x1').plot()
df.sort_values(by='x1').plot()

plt.show() # fix this

# 2.3

# --__('-')__--
#       |
#     _/ \_
