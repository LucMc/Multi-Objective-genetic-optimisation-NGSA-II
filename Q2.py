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
import sys

# Set random seed and options for printing pandas
# np.random.seed(1)
# random.seed(1)
# torch.manual_seed(1)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

population = 100 # Population for each gen
bit_length = 15
numOfBits = 15*6 # Number of bits in the chromosomes
maxnum = 2**bit_length # absolute max size of number coded by binary list 1,0,0,1,1,....
flip_prob = 1/(numOfBits*6) # probability of bit flipping
cross_prob = 0.9
NGEN = 60 # number of generations
# torch.set_default_dtype(torch.float32)

# Function for net approximation
def f(x1, x2):
    return sin(2*(x1) + 2.0) * cos(0.5 * x2) + 0.5

# Generate initial data Q2.2
def generate_data():
    df = pd.DataFrame(columns=['x1', 'x2', 'output'])
    for i in range(25):
        x1 = random.uniform(0, 2*pi)
        x2 = random.uniform(0, 2*pi)

        y = f(x1, x2)
        df.loc[i] = [x1, x2, y]

    # print(df.head())
    train = df[:11] # 11
    test = df[11:] # 11
    # print(train)
    # print(test)

    # Save dataframes as .dat files as space seperated values
    print('Saving dat files.')
    train.to_csv('./train.dat', index=False, sep=' ', header=False)
    test.to_csv('./test.dat', index=False, sep=' ', header=False)
    return df, test, train

# Used to 3D plot x1 and x2 with output
def plot(*args):
    for df in args:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['x1'].to_numpy(), df['x2'].to_numpy(), df['output'].to_numpy())
        plt.show() # for line use plot3D

# Plot a generation
def plot_gen(df, initial_df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(initial_df['f1'], initial_df['f2'], s=20, c='b', marker="o", label='initial generation')
    ax1.scatter(df['f1'], df['f2'], s=20, c='r', marker="o", label='next generation')
    plt.legend(loc='upper left');
    plt.show()

# train.sort_values(by='x1').plot()
# test.sort_values(by='x1').plot()
# df.sort_values(by='x1').plot()
#
# plt.show() # fix this


# Neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(2, 6, bias=False) # 2 Inputs to hidden layer
        self.out = nn.Linear(6, 1, bias=False) # 1 output

    # Forward pass
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x)) # Sigmoid activation on hidden layer
        x = self.out(x) # Linear activation for output
        return x # F.log_softmax(x, dim=0)

# generate data
df, test, train = generate_data()

data = genfromtxt('train.dat', delimiter=' ') # change back to test 1, test 2 shows more samples

# Format training dataset
x_train = data[:, 0:2]
y_train = data[:, 2]
x_train = torch.as_tensor(x_train, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)

# Format testing dataset
data = genfromtxt('test.dat', delimiter=' ')
x_test = data[:, 0:2]
y_test = data[:, 2]
x_test = torch.as_tensor(x_test, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32)


'''
print("\nWeights:")
print (net.out.weight)
print (net.hidden.weight)
net.out.weight[0][0]=1
print (net.out.weight)
net.out.weight = torch.nn.parameter.Parameter(torch.as_tensor([[ 1.0000,  0.2965,  0.1628,  0.6232, -1.0667, -2.1927]]))

print (net.out.weight)
'''

# print("\nNet: \n", net)  # net architecture

# Perhaps use torch.no_grad in the first case before lifetime learning

def train_net(df, net, method='Lamarckian'):
    ##
    if method == 'Lamarckian':
        method = 'Baldwinian'

    optimizer = torch.optim.Rprop(net.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    for ind, row in df.iterrows():
        # print(f"Before: {net.out.weight}")
        net = Net() # Reset network for new weights

        # initialise weights

        weights = [float(x) for x in row.drop('MSE').apply(chrom_to_real).to_numpy()]
        net.out.weight = torch.nn.parameter.Parameter(torch.as_tensor([weights]))
        # print(f"FIRST WEIGHT: {net.out.weight}")

        if method == 'Lamarckian':
            df.loc[ind]['MSE'] = test_loss(net) # Loss determined

        for i in range(len(x_train)):
            # data is a batch of featuresets and labels
            X, y = x_train[i], y_train[i]
            net.zero_grad()
            output = net(X)
            # print(output)

            # loss = F.nll_loss(output, y)
            # print("start: ", net.out.weight)

            loss = loss_fn(output, y.view(1))
            loss.backward()
            optimizer.step() # part 2.7 RPROP
            # print("Changed", net.out.weight, "\n\n")

            # update table weights
            # print(df[df.index == ind])
        # print(total_loss)
        # print(f"After: {net.out.weight}")

        # Selection pressure is present in both

        # Validation
        '''Is selection pressure preasent in both?'''

        if method != 'Baldwinian' and method != 'Lamarckian':
            print("Please Choose an evolution strategy.")
            sys.exit()

        elif method == 'Lamarckian':
            # Lamarckian
            # change weights in dataframe
            df.loc[ind][['w1', 'w2', 'w3', 'w4', 'w5', 'w6']] = [real_to_chrom(x) for x in net.out.weight[0]]
        else:
            # Baldwinian
            df.loc[ind]['MSE'] = test_loss(net)


    return df


''' Testing '''
def test_loss(net):
    # Mean Squared Error as a loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')
    total_loss = 0
    # Don't want to train on testing data
    with torch.no_grad():
        # for each row in testing data
        for i in range(len(x_test)):
            # define input and expected output
            X, y = x_test[i], y_test[i]
            # predict using given model
            output = net(X)
            # Calculate loss
            loss = loss_fn(output, y.view(1))
            total_loss += loss.item()
    return total_loss

# Convert chromosome into real
def chrom_to_real(c):
    indasstring=''.join(map(str, c)) # convert list to string
    numasint=int(indasstring, 2) # convert to int from base 2 list
    numinrange = -10 + 20*numasint/maxnum # CHANGED TO OUR VALUES
    # print(numinrange)
    return numinrange

def real_to_chrom(r):
    # reverse process of chrom_to_real
    numasint = int(((r + 10)/20)*maxnum)
    indasstring = str(bin(numasint))[2:]
    for _ in range(15 - len(indasstring)):
        indasstring = '0' + indasstring
    chrom = list(indasstring)
    # print(chrom)
    return ''.join(chrom)

# print(chrom_to_real(real_to_chrom(5)))
# sys.exit()

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
        return selection[selection['MSE'] == min(selection['MSE'])]

    # Uniform Crossover
    def uniform(indv1, indv2):
        # Recombine individual chromosomes
        individual1_chromosome = "" + indv1['w1'].item() + indv1['w2'].item() + indv1['w3'].item() + \
                                 indv1['w4'].item() + indv1['w5'].item() + indv1['w6'].item()
        individual2_chromosome = "" + indv2['w1'].item() + indv2['w2'].item() + indv2['w3'].item() + \
                                 indv2['w4'].item() + indv2['w5'].item() + indv2['w6'].item()

        if random.random() < cross_prob: # Chance of crossover
            for i in range(len(individual1_chromosome)): # for every bit in the chromosome length
                if random.choice([0, 1]) == 1: # Randomly choose weather child has indv1 or indv2's bit
                    individual1_chromosome = individual1_chromosome[:i] + individual2_chromosome[i] + individual1_chromosome[i+1:]
                    individual2_chromosome = individual2_chromosome[:i] + individual1_chromosome[i] + individual2_chromosome[i+1:]

                # Random bit flip chance for each chromosome
                if random.random() < flip_prob:
                    individual1_chromosome = individual1_chromosome[:i] + str(int(individual1_chromosome[i]) ^ 1) + individual1_chromosome[i+1:]
                if random.random() < flip_prob:
                    individual2_chromosome = individual2_chromosome[:i] + str(int(individual2_chromosome[i]) ^ 1) + individual2_chromosome[i+1:]

        return individual1_chromosome, individual2_chromosome

    # Tournament selection twice to create 2 children
    parent1 = sample_pair(df)
    parent2 = sample_pair(df)
    # Uniform crossover on children
    child1, child2 = uniform(parent1, parent2)
    return child1, child2

# Selection and crossover to create a new generation
def next_generation(df):
    # Initialise new generation
    next_gen_df = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'MSE'])
    index = 0 # Index in DataFrame
    for i in range(int(population/2)):# for range of half because selection and crossover produces 2 children
        children = tournament_selection(df)
        # for each child
        for child in children:
            # Weight variables determined from chromosome
            w1 = child[:bit_length]
            w2 = child[bit_length:bit_length * 2]
            w3 = child[bit_length * 2:bit_length * 3]
            w4 = child[bit_length * 3:bit_length * 4]
            w5 = child[bit_length * 4:bit_length * 5]
            w6 = child[bit_length * 5:]

            # Append weight variables into DataFrame
            next_gen_df.loc[index] = [w1, w2, w3, w4, w5, w6, 1]
            index += 1

    return next_gen_df

# Main function for our program
def main():
    initial_net = Net()  # define the network
    # Q2.2
    df = generateDataFrame() # Initialise weights

    # Using Balwinian we can initialise the MSE without changing weights.
    # This net is only used as an initial starting point
    train_net(df, initial_net, method='Baldwinian') # Initialise MSE fitness but dont change genes

    # Q2.3 2.3 Binary GA optimisor

    # Print the initial DataFrame
    print(df)
    print(df.mean())

    initial = df
    # Plot generations
    generations = {}
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # Genetic optimisation comparison
    print("START OF GENERATIONS\n\n")
    # Initialise Lamarckian and Baldwinian networks
    lnet = Net()
    bnet = Net()

    for method in ['Lamarckian', 'Baldwinian']:
        df = initial
        net = Net()  # re-initialise network
        # print(df)
        if method == 'Lemarkian':
            cmap = 'BuPu'
            nn = lnet
        else:
            cmap = 'PuRd'
            nn = bnet

        # initialise for first gen
        generations[0] = min(df['MSE'])

        for i in range(1, NGEN):
            df = train_net(next_generation(df), method=method, net=net)
            # print(df['MSE'])
            print(df['MSE'].mean())

            generations[i] = min(df['MSE'])
            print(min(df['MSE']))
            # print('TESTINGTEST')
            # print(generations[i])
            # print(df)
            # test_net()
        print(list(generations.keys()), generations.values())
        ax1.scatter(list(generations.keys()), list(generations.values()), alpha=0.9, cmap=cmap, label=method)

    plt.legend(loc='upper left')
    plt.show()

    # plot(train, test) # Keep this in normally

    # starting off by using normal optimisor


if __name__ == '__main__':
    main()

# --__('-')__--
#       |
#     _/ \_

'''
##
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
    plt.show()

    print(f'accuracy over random: {compare} / {total}')
    return compare, total
    
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
