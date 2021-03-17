import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.modules.activation import PReLU
import time

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

def get_data(path):
    df = pd.read_csv(path)

    df = df.drop(["MinTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"
    ],axis = 1)

    df = df.dropna()

    temp = ""
    locations = []
    for i in df["Location"]:
        if not i == temp:
            temp = i
            locations.append(df[df["Location"]==i])
            if len(locations) == 15:
                break

    X = []
    Y = []

    for i in range(len(locations)):
        for j in range(len(locations[i]["MaxTemp"].to_numpy())-1):
            X.append([locations[i]["MaxTemp"].to_numpy()[j],i])
            Y.append(locations[i]["MaxTemp"].to_numpy()[j+1])

    np_X = np.array(X)
    np_Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(np_X,np_Y,test_size=0.25)

    return X_train, X_test, Y_train, Y_test


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
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_fn = torch.nn.MSELoss(reduction='mean')

for epoch in range(100):
    
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

"""
output of code:
0: 597.5784301757812
1: 569.7161865234375
2: 543.0737915039062
3: 516.3881225585938
4: 487.4836730957031
5: 455.2174377441406
6: 419.3431091308594
7: 379.6042175292969
8: 336.7672119140625
9: 291.39910888671875
10: 244.1245880126953
11: 196.51426696777344
12: 150.80419921875
13: 110.19776153564453
14: 79.15937805175781
15: 62.941802978515625
16: 65.78953552246094
17: 86.09563446044922
18: 111.6891860961914
19: 127.39710235595703
20: 127.3395767211914
21: 115.0817642211914
22: 97.60432434082031
23: 81.02761840820312
24: 69.01788330078125
25: 62.72466278076172
26: 61.51613235473633
27: 63.85298538208008
28: 67.98210906982422
29: 72.38742065429688
30: 75.98523712158203
31: 78.14140319824219
32: 78.63375091552734
33: 77.53526306152344
34: 75.15058898925781
35: 71.93423461914062
36: 68.42699432373047
37: 65.18250274658203
38: 62.68522262573242
39: 61.263038635253906
40: 61.0110969543457
41: 61.75495910644531
42: 63.08408737182617
43: 64.46566009521484
44: 65.41001892089844
45: 65.62635040283203
46: 65.09425354003906
47: 64.02884674072266
48: 62.769508361816406
49: 61.64762878417969
50: 60.88832473754883
51: 60.57012176513672
52: 60.63882064819336
53: 60.95491027832031
54: 61.34702682495117
55: 61.6646728515625
56: 61.80413055419922
57: 61.72394943237305
58: 61.450008392333984
59: 61.06000900268555
60: 60.65055847167969
61: 60.313697814941406
62: 60.11417770385742
63: 60.07181930541992
64: 60.15362548828125
65: 60.283729553222656
66: 60.37706756591797
67: 60.37899398803711
68: 60.2985954284668
69: 60.12715148925781
70: 59.950965881347656
71: 59.79722595214844
72: 59.69233703613281
73: 59.64418029785156
74: 59.63499450683594
75: 59.640567779541016
76: 59.63839340209961
77: 59.61436462402344
78: 59.565673828125
79: 59.50041198730469
80: 59.43229293823242
81: 59.37521743774414
82: 59.33769226074219
83: 59.319766998291016
84: 59.312984466552734
85: 59.30467987060547
86: 59.285213470458984
87: 59.2526969909668
88: 59.21234130859375
89: 59.1722526550293
90: 59.139122009277344
91: 59.11541748046875
92: 59.099029541015625
93: 59.08515167236328
94: 59.06892395019531
95: 59.047637939453125
96: 59.02156448364258
97: 58.99320602416992
98: 58.96574401855469
99: 58.941463470458984
mean error in train set: 58.941463470458984
mean error in test set: 58.48478317260742
done in seconds: 18.43150758743286
"""