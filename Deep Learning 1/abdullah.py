# Ben bir sey anlamadim bundan

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.fc4 = nn.Linear(dims[3], dims[4])
        self.fc5 = nn.Linear(dims[4], dims[5])
        self.fc6 = nn.Linear(dims[5], dims[6])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)

dic = dict()
data = pd.read_csv("weatherAUS.csv")[['Location', 'MaxTemp']]
data = data.dropna()
for city in set(set(data['Location'])):
    each_city = data.loc[data['Location'] == city]
    each_city = each_city.set_index([pd.Index(list(range(1, each_city.shape[0] + 1)))])
    dic[city] = each_city

city = "Portland"  # Select a city
Y = torch.tensor(dic[city]["MaxTemp"].values, device=device, dtype=dtype)
X = torch.arange(1, Y.shape[0] + 1, device=device, dtype=dtype)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=34)
X_train.unsqueeze_(-1)
X_test.unsqueeze_(-1)

net = Net(dims=[1, 10, 20, 20, 20, 10, 1])
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss(reduction='mean')

for epoch in range(1000):
    output = net(X_train)
    output.size = Y_train.size
    loss = loss_fn(output, Y_train)
    print(str(epoch) + ": " + str(loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("mean error in train set: " + str(loss.item()))

pred = net(X_test)
pred.size = Y_test.size
loss = loss_fn(pred, Y_test)
print("mean error in test set: " + str(loss.item()))

res = [[i, j, k] for i, j, k in zip(X_test.cpu(), Y_test.cpu(), pred.cpu().detach())]
sorted_list = sorted(res, key=lambda x: x[0])
x = [i[0] for i in sorted_list]
y = [i[1] for i in sorted_list]
p = [i[2] for i in sorted_list]
plt.scatter(x, y,  color='black', linewidth=1)
plt.plot(x, p, color='blue', linewidth=2)
plt.show()
