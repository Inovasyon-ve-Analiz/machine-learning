import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import time
import sys
from get_data import get_data

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(49, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 5)
        self.fc5 = nn.Linear(5, 1)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

def train(X_train, Y_train, model, loss_fn, optimizer):
    size = len(Y_train)
    train_loss = 0
    for batch, (X, y) in enumerate(zip(X_train, Y_train)):
        X = X.view(1,49)
        y = y.view(1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss_fn(pred, y).item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= size
    print(f"Train error: \nAvg loss: {train_loss:>8f}\n")

def test(X_test, Y_test, model, loss_fn):
    size = len(Y_test)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in zip(X_test, Y_test):
            X = X.view(1,49)
            y = y.view(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")

if __name__ =="__main__":

    path = "weatherAUS.csv"
    run_on_gpu = True

    X_train, X_test, Y_train, Y_test = get_data(path)

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    Y_train = torch.from_numpy(Y_train).float()
    Y_test = torch.from_numpy(Y_test).float()
    tic = time.time()
    net = Net()
    if run_on_gpu:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    loss_fn = nn.MSELoss(reduction='mean')

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-----------------------------")
        train(X_train, Y_train, net, loss_fn, optimizer)
        test(X_test, Y_test, net, loss_fn)
        

    print(f"done in {time.time()-tic} seconds")
