import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from random import randrange
from get_data import get_data

X_train, X_test, Y_train, Y_test = get_data("weatherAUS.csv")
#print(X_train.shape, X_test.shape)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
Y_train = torch.from_numpy(Y_train).float()
Y_test = torch.from_numpy(Y_test).float()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(49, 100)
        self.l2 = nn.Linear(100, 70)
        self.l3 = nn.Linear(70, 40)
        self.l4 = nn.Linear(40, 10)
        self.l5 = nn.Linear(10, 1)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out = F.relu(self.l4(out))
        # since it is a numerical estimation, by prof ng's advice:
        # no activation functions were used on the output layer
        out = self.l5(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet()
model.to(device)

num_epochs = 100
learning_rate = 0.001

# loss and optimizer
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    X_train = X_train.to(device)
    labels = Y_train.to(device)
    new_shape = (len(labels), 1)
    labels = labels.view(new_shape)

    # forward prop
    outputs = model(X_train)
    loss = criteria(outputs, labels)

    # backward prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1} over {num_epochs}, step {epoch + 1} over {num_epochs}, loss = {loss.item():.4f}')

with torch.no_grad():
    X_test = X_test.to(device)
    labels = Y_test.to(device)
    outputs = model(X_test)
    random_selection = randrange(0, len(outputs) - 1)
    #print(random_selection)
    print('prediction:', "{:.2f}".format(outputs[random_selection].item()))
    print('real value:', "{:.2f}".format(labels[random_selection].item()))
    difference = outputs[random_selection] - labels[random_selection]
    print('difference: ', "{:.2f}".format(difference.item()))
