import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

loss_fn = torch.nn.MSELoss(reduction='mean')

for epoch in range(10000):
    
    output = net(X_train)
    output.size = Y_train.size
    loss = loss_fn(output, Y_train)
    print(str(epoch)+": "+str(loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("mean error in train set: " +str(loss.item()))

pred = net(X_test)
pred.size = Y_test.size
loss = loss_fn(pred, Y_test)
print("mean error in test set: " +str(loss.item()))
toc = time.time()
print("done in seconds: "+str(toc-tic))
