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

random.seed(1)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

population = 100
bit_length = 15
numOfBits = 15 # Number of bits in the chromosomes
maxnum = 2**numOfBits # absolute max size of number coded by binary list 1,0,0,1,1,....
flip_prob = 0.9
NGEN = 30
# torch.set_default_dtype(torch.float32)


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

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 6) # input (image) // output
        self.out = nn.Linear(6, 1) # input (image) // output

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x # F.log_softmax(x, dim=0)

df, test, train = generate_data()

data = genfromtxt('train2.dat', delimiter=' ') # change back to test 1, test 2 shows more samples
x_train = data[:, 0:2]
y_train = data[:, 2]
x_train = torch.as_tensor(x_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)

net = Net()  # define the network

data = genfromtxt('test2.dat', delimiter=' ')
x_test = data[:, 0:2]
y_test = data[:, 2]
x_test = torch.as_tensor(x_test, dtype=torch.float32)

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

def train_net(df):
    # What we want to optimise // learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-3) # this is what we want to do with GA
    total_loss = 0
    EPOCHS = 1
    loss_fn = torch.nn.MSELoss(reduction='sum')
    for epoch in range(EPOCHS):
        # ignore epoch for now
        for ind, row in df.iterrows():
            # initialise weights
            # print(row.drop('MSE').apply(chrom_to_real).to_numpy())
            weights = [float(x) for x in row.drop('MSE').apply(chrom_to_real).to_numpy()]
            net.out.weight = torch.nn.parameter.Parameter(torch.as_tensor([weights]))

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
                # optimizer.step()
                total_loss += loss.item()
                # update table weights
                # print(df[df.index == ind])
            # print(total_loss)
            df.loc[ind]['MSE'] = total_loss
            total_loss = 0
            # print(loss.item())
    return df


''' Testing '''


def test_net():
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

    # For some reason it predicts but not extreme enough
    plt.plot([x for x in results], color="red")
    plt.plot(ans)
    # plt.show()

    print(f'accuracy over random: {compare} / {total}')
    return compare, total


# y_test = torch.as_tensor(y_train, dtype=torch.float32)

# x_test = [1.818473210508727, 3.356242679806961]






'''
TODO
copy GA []
Weights between -10 and 10 as decision variables (6) at 15 bits each []
minimise loss []
how many bits for encoding weights total? 15 x 6 = 90 
get GA done by christmas

then loads of other parts:(

'''

def chrom_to_real(c):
    indasstring=''.join(map(str, c))
    # degray=gray_to_bin(indasstring)
    # print(indasstring)
    numasint=int(indasstring, 2) # convert to int from base 2 list
    numinrange = -10 + 20*numasint/maxnum # CHANGED TO OUR VALUES
    # print(numinrange)
    return numinrange


def generateDataFrame():
    df = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'MSE'])

    for i in range(population):
        individual = [random.randint(0, 1) for _ in range(bit_length*6)]
        # Change this so its one long variable that is split up
        individual = [str(x) for x in individual]
        w1 = "".join(individual[:bit_length])
        w2 = "".join(individual[bit_length:bit_length*2])
        w3 = "".join(individual[bit_length*2:bit_length*3])
        w4 = "".join(individual[bit_length*3:bit_length*4])
        w5 = "".join(individual[bit_length*4:bit_length*5])
        w6 = "".join(individual[bit_length*5:])

        # df.loc[i] = [chrom_to_real(w1), chrom_to_real(w2), chrom_to_real(w3), chrom_to_real(w4), chrom_to_real(w5), chrom_to_real(w6), MSE(w1, w2, w3, w4, w5, w6)]
        df.loc[i] = [w1, w2, w3, w4, w5, w6, 1] # 1 as a placeholder for MSE
    return df

def tournament_selection(df):
    # Chose two random rows
    def sample_pair(df):
        # Tournament Selection
        selection = df.sample(n=2)
        print(selection)

        return selection[selection['MSE'] == min(selection['MSE'])]

    def uniform(indv1, indv2):
        # Crossover
        individual1_chromosome = "" + indv1['w1'].item() + indv1['w2'].item() + indv1['w3'].item() + \
                                 indv1['w4'].item() + indv1['w5'].item() + indv1['w6'].item()
        individual2_chromosome = "" + indv2['w1'].item() + indv2['w2'].item() + indv2['w3'].item() + \
                                 indv2['w4'].item() + indv2['w5'].item() + indv2['w6'].item()

        if random.random() < 0.9: # crossover probability
            for i in range(len(individual1_chromosome)):
                if random.random() < 0.5: # This needs to be inversely proportional maybe
                    individual1_chromosome = individual1_chromosome[:i] + individual2_chromosome[i] + individual1_chromosome[i+1:]
                    individual2_chromosome = individual2_chromosome[:i] + individual1_chromosome[i] + individual2_chromosome[i+1:]

                if random.random() < flip_prob:
                    individual1_chromosome = individual1_chromosome[:i] + str(int(individual1_chromosome[i]) ^ 1) + individual1_chromosome[i+1:]
                if random.random() < flip_prob:
                    individual2_chromosome = individual2_chromosome[:i] + str(int(individual2_chromosome[i]) ^ 1) + individual2_chromosome[i+1:]

        return individual1_chromosome, individual2_chromosome

    parent1 = sample_pair(df)
    parent2 = sample_pair(df)
    child1, child2 = uniform(parent1, parent2)
    return child1, child2

def next_generation(df):
    next_gen_df = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'MSE'])
    index = 0
    # for gen in range(20):
    for i in range(int(population/2)):
        children = tournament_selection(df)
        for child in children:
            print(child)
            w1 = child[:bit_length]
            w2 = child[bit_length:bit_length * 2]
            w3 = child[bit_length * 2:bit_length * 3]
            w4 = child[bit_length * 3:bit_length * 4]
            w5 = child[bit_length * 4:bit_length * 5]
            w6 = child[bit_length * 5:]
            #
            # x1 = child[:bit_length]
            # x2 = child[bit_length:bit_length * 2]
            # x3 = child[bit_length*2:]
            #
            # _f1 = f1(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))
            # _f2 = f2(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))

            # next_gen_df.loc[index] = [x1, x2, x3, _f1, _f2]
            # print(len(w1), len(w2), len(w3), len(w4), len(w5), len(w6))
            next_gen_df.loc[index] = [w1, w2, w3, w4, w5, w6, 1]
            index += 1

    return next_gen_df
'''
Now add new MSE for next generation then BAM we nearly there
'''
def main():
    # Q2.2
    #df, test, train = generate_data()
    df = generateDataFrame()
    train_net(df)
    # test_net()
    print(df)
    # Genetic optimisation
    print(next_generation(df))
    # plot(train, test) # Keep this in normally

    # Q2.3 2.3 Binary GA optimisor
    # starting off by using normal optimisor


if __name__ == '__main__':
    main()

# --__('-')__--
#       |
#     _/ \_

'''
##
MSE(chrom_to_real(w1),
                                                 chrom_to_real(w2),
                                                 chrom_to_real(w3),
                                                 chrom_to_real(w4),
                                                 chrom_to_real(w5),
                                                 chrom_to_real(w6))
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
