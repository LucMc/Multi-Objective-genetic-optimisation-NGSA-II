import torch
import torch.nn.functional as F
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np

# load the training data
data = genfromtxt('cancer_TR.dat', delimiter=' ')
x = data[:, 0:9]
y = data[:, 9:11]
x = torch.as_tensor(x, dtype=torch.float32)
y = torch.as_tensor(y, dtype=torch.float32)


print(x.size())
print(y)


# torch.manual_seed(1)    # reproducible experiments
                          # by fixing the seed you will remove randomness
# set up the network
class Net(torch.nn.Module):
    # initialise one hidden layer and one output layer
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    # connect up the layers: the input passes through the hidden, then the sigmoid, then the output layer
    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.out(x)
        return x


net = Net(n_feature=9, n_hidden=5, n_output=2)  # define the network
print(net)  # net architecture

'''Train'''
optimizer = torch.optim.Rprop(net.parameters(), lr=0.02)
    # SGD or Rprop
loss_func = torch.nn.MSELoss()
running_loss = 0.0
loss_values = []


'''Test'''
# load the test data
test = genfromtxt('cancer_tt.dat', delimiter=' ')
x2 = test[:, 0:9]
y2 = test[:, 9:11]
x2 = torch.as_tensor(x, dtype=torch.float32)
y2 = torch.as_tensor(y, dtype=torch.float32)


_, target_indices = torch.max(y2, 1)
total = y2.size(0)
accuracy = 0



for rep in range(10):
    '''Train'''
    net.hidden.reset_parameters()
    net.out.reset_parameters()
#     for layer in net.children():
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()
    for t in range(60):
        out = net(x)  # input x and predict based on x
        loss = loss_func(out, y)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        loss_values.append(loss.item()) # keep track of loss values for later plot

    _, predicted_indices = torch.max(net(x2), 1)
    accuracythistime = (predicted_indices == target_indices ).sum().item() / total
    print("accuracy ",rep,":", accuracythistime)

    accuracy += accuracythistime

accuracy/=10


print("accuracy:", accuracy)