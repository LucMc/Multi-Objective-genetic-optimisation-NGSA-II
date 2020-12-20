#import torch

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
from mpl_toolkits import mplot3d
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy import genfromtxt


os.environ['KMP_DUPLICATE_LIB_OK']='True'


def f(x1, x2):
    return sin(2*(x1) + 2.0) * cos(0.5 * x2) + 0.5


def generate_data():
    df = pd.DataFrame(columns=['x1', 'x2', 'output'])
    for i in range(300):
        x1 = random.uniform(0, 2*pi)
        x2 = random.uniform(0, 2*pi)

        y = f(x1, x2)
        df.loc[i] = [x1, x2, y]

    # print(df.head())
    train = df[:200] # 11
    test = df[200:] # 11
    print(train)
    print(test)

    #if not os.path.isfile('./train2.dat') or not os.path.isfile('./test2.dat'):
    print('Saving dat files.')
    train.to_csv('./train2.dat', index=False, sep=' ', header=False)
    test.to_csv('./test2.dat', index=False, sep=' ', header=False)
    return df, test, train

def plot(*args):
    for df in args:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['x1'].to_numpy(), df['x2'].to_numpy(), df['output'].to_numpy())
        plt.show() # for line use plot3D


# train.sort_values(by='x1').plot()
# test.sort_values(by='x1').plot()
# df.sort_values(by='x1').plot()
#
# plt.show() # fix this


# load the training data
df, test, train = generate_data()

data = genfromtxt('train2.dat', delimiter=' ')
x_train = data[:, 0:2]
y_train = data[:, 2]
x_train = torch.as_tensor(x_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 6) # input (image) // output
        self.out = nn.Linear(6, 1) # input (image) // output

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x # F.log_softmax(x, dim=0)

net = Net()  # define the network

'''
print("\nWeights:")
print (net.out.weight)
print (net.hidden.weight)
net.out.weight[0][0]=1
print (net.out.weight)
net.out.weight = torch.nn.parameter.Parameter(torch.as_tensor([[ 1.0000,  0.2965,  0.1628,  0.6232, -1.0667, -2.1927]]))

print (net.out.weight)
'''

print("\nNet: \n", net)  # net architecture

# What we want to optimise // learning rate
optimizer = optim.Adam(net.parameters(), lr=1e-3) # this is what we want to do with GA

EPOCHS = 10
loss_fn = torch.nn.MSELoss(reduction='sum')
for epoch in range(EPOCHS):
    for i in range(len(x_train)):
        # print(i, x_train, y_train)
        # data is a batch of featuresets and labels
        X, y = x_train[i], y_train[i]
        # print(X, y)
        net.zero_grad()
        output = net(X) # .unsqueeze(dim=0)
        # loss = F.nll_loss(output, y)
        loss = loss_fn(output, y.view(1))

        # print(loss.item())
        loss.backward()
        optimizer.step()
    print(loss.item())
    if loss.item() < 0.001:
        print(epoch)
        break

data = genfromtxt('test2.dat', delimiter=' ')
x_test = data[:, 0:2]
y_test = data[:, 2]
x_test = torch.as_tensor(x_test, dtype=torch.float32)
# y_test = torch.as_tensor(y_train, dtype=torch.float32)

# x_test = [1.818473210508727, 3.356242679806961]

compare = 0
total = 0
results = []
ans = []
for i in range(len(x_test)-1):
    y_pred_to_validate = net(x_test[i])
    # print(y_pred_to_validate.detach().numpy())
    # print(y_test[i])
    num = random.uniform(f(0, 0), f(2*pi, 2*pi))
    results.append(y_pred_to_validate.detach().numpy()[0])
    ans.append(y_test[i])
    # print(num, y_test[i], y_pred_to_validate.detach().numpy()[0], '\n\n')
    if abs(num - y_test[i]) > abs(y_pred_to_validate.detach().numpy()[0] - y_test[i]):
        compare += 1
    total += 1

print(f'accuracy over random: {compare} / {total}')

# For some reason it predicts but not extreme enough
plt.plot([x for x in results], color="red")
plt.plot(ans)
plt.show()

'''
TODO
copy GA []
Weights between -10 and 10 as decision variables (6) at 15 bits each []
minimise loss []
how many bits for encoding weights total? 15 x 6 = 90 
get GA done by christmas

then loads of other parts:(

'''
population = 100
bit_length = 15
numOfBits = 15*6 # Number of bits in the chromosomes
maxnum = 2**numOfBits # absolute max size of number coded by binary list 1,0,0,1,1,....
flip_prob = 0.9
NGEN = 30

def chrom_to_real(c):
    indasstring=''.join(map(str, c))
    # degray=gray_to_bin(indasstring)
    numasint=int(indasstring, 2) # convert to int from base 2 list
    numinrange = -4+8*numasint/maxnum # CHANGED TO OUR VALUES
    # print(numinrange)
    return numinrange

def generateDataFrame():
    df = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6'])

    for i in range(population):
        individual = [random.randint(0, 1) for _ in range(bit_length*6)]
        # Change this so its one long variable that is split up
        individual = [str(x) for x in individual]
        w1 = "".join(individual[:15])
        w2 = "".join(individual[15:30])
        w3 = "".join(individual[30:45])
        w4 = "".join(individual[45:60])
        w5 = "".join(individual[60:75])
        w6 = "".join(individual[90:])


        _f1 = f1(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))
        _f2 = f2(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))

        df.loc[i] = [x1, x2, x3, _f1, _f2]

        # df.to_pickle('testing.pickle')
        # df = pd.read_pickle('testing.pickle')
    return df

def main():
    # Q2.2
    df, test, train = generate_data()
    # plot(train, test) # Keep this in normally

    # Q2.3 2.3 Binary GA optimisor
    # starting off by using normal optimisor




# if __name__ == '__main__':
#     main()

# --__('-')__--
#       |
#     _/ \_

'''
##

##
# net = torch.nn.Sequential(
#     torch.nn.Linear(2, 200),
#     torch.nn.LeakyReLU(),
#     torch.nn.Linear(200, 100),
#     torch.nn.LeakyReLU(),
#     torch.nn.Linear(100, 1),
# )
##

# print(y_pred_to_validate.detach().numpy())
# 'TRAIN
# optimizer = torch.optim.Rprop(net.parameters(), lr=0.02)
# # SGD or Rprop
# loss_func = torch.nn.MSELoss()
# running_loss = 0.0
# loss_values = []
# ##
#
# ##
#
# TEST
# # load the test data
# test = genfromtxt('test.dat', delimiter=' ')
# x2 = test[:, 0:1]
# y2 = test[:, 2]
# x2 = torch.as_tensor(x, dtype=torch.float32)
# y2 = torch.as_tensor(y, dtype=torch.float32)
#
# print(y2.size(), x2.size(), x.size(), y.size())
# _, target_indices = torch.max(y2, 0)
# total = y2.size(0)
# accuracy = 0
#
#
#
# for rep in range(10):
#     #TRAIN
#     net.fc1.reset_parameters()
#     net.fc2.reset_parameters()
# #     for layer in net.children():
# #         if hasattr(layer, 'reset_parameters'):
# #             layer.reset_parameters()
#     for t in range(60):
#         out = net(x)  # input x and predict based on x
#         loss = loss_func(out, y)
#         optimizer.zero_grad()  # clear gradients for next train
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients
#         loss_values.append(loss.item()) # keep track of loss values for later plot
#
#     _, predicted_indices = torch.max(net(x2), 1)
#     accuracythistime = (predicted_indices == target_indices ).sum().item() / total
#     print("accuracy ",rep,":", accuracythistime)
#
#     accuracy += accuracythistime
#
# accuracy/=10
#
#
# print("accuracy:", accuracy)
#

##
# old net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 6) # input (image) // output
        self.fc2 = nn.Linear(6, 6) # input (image) // output
        self.fc3 = nn.Linear(6, 1) # input (image) // output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    net = Net()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # this is what we want to do with GA
    EPOCHS = 3

    for epoch in range(EPOCHS):
        for data in train:
            # data is a batch of featuresets and labels
            X, y = data['x1'], data['y'] # split into input and output columns
            net.zero_grad()
            output = net(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

# other
EPOCHS = 3

for epoch in range(EPOCHS):
    for i in range(11):
        # data is a batch of featuresets and labels
        X, y = x[i], y[i]
        net.zero_grad()
        output = net(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)



'''
