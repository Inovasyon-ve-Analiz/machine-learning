import torch
from torch import nn
from torch.nn import functional as F

import cv2

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,3)
        
        self.fc1 = nn.Linear(5*5*16,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,10)
  
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
    
        x = x.view(-1,5*5*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)

        return x

model = MyNet()
model.load_state_dict(torch.load("model.pt"))
model = model.cuda()

img8 = cv2.imread("8.png",0)
img6 = cv2.imread("6.png",0)

data6 = torch.reshape(torch.tensor(img6),(1,1,28,28)).float()
target6 = torch.tensor(6)

data8 = torch.reshape(torch.tensor(img8),(1,1,28,28)).float()
target8 = torch.tensor(8)

model.eval()
correct = 0
with torch.no_grad():

    data, target = data6.cuda(), target6.cuda()
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()

print(correct)
print(pred)