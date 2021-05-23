import torch 
from torchvision import models
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

import time

def train(model, train_loader, optimizer, epoch):
    model.train()
    new_target = torch.zeros([10,10])
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.long()
        for i in range(10):
            new_target[i] = torch.eye(10)[target[i]]
        new_target = new_target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, new_target)
      
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    new_target = torch.zeros([10,10])
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.long()
            for i in range(10):
                new_target[i] = torch.eye(10)[target[i]]
            new_target = new_target.cuda()
            target = target.cuda()
            output = model(data)
            test_loss += F.mse_loss(output, new_target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


resnet = models.resnet18()

resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.fc = nn.Linear(in_features=512, out_features=10, bias=True)

resnet = resnet.cuda()

train_MNIST = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_MNIST = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_set = DataLoader(train_MNIST, batch_size=10, shuffle=True)
test_set = DataLoader(test_MNIST, batch_size=10, shuffle=True)

optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

epochs = 3
for epoch in range(epochs):
    train(resnet,test_set,optimizer,epoch)
    test(resnet,test_set)
    time.sleep(3)