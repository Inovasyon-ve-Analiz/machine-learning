import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

# hyper-parameters
input_size = 784  # 28*28 pixel images
batch_size = 100
num_epochs = 2
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                           transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                          shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # MNIST consists of gray-scale images, so channel is 1
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 28 - 3 + 1 = 26
        self.pool1 = nn.MaxPool2d(2, 2)
        # (26 - 2) / 2 + 1 = 13
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 13 - 3 + 1 = 11
        self.pool2 = nn.MaxPool2d(5, 1)
        # pool2: 11 - 5 + 1 = 7
        self.fc1 = nn.Linear(16*7*7, 80) # 16*7*7
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        #print(out.shape)
        out = out.view(-1, 16*7*7)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


model = NeuralNet()
model.to(device)

# loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward prop
        outputs = model(images)
        loss = criteria(outputs, labels)

        # backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} over {num_epochs}, step {i + 1} over {n_total_steps}, loss = {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * (n_correct / n_samples)
    print(f'accuracy: {acc}')
