import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

a = 100
drones = np.ndarray((1,a**2))
for filename in glob.glob('../dataset/drone/*.jpg'):
    img = cv2.resize(cv2.imread(filename,0),(a,a)).reshape(1,a**2)[0]
    drones = np.vstack((drones,img))  
drones = drones[1:]

antis = np.ndarray((1,a**2))
for filename in glob.glob('../dataset/anti_drone/*.jpg'):
    img = cv2.resize(cv2.imread(filename,0),(a,a)).reshape(1,a**2)[0]
    antis = np.vstack((antis,img))  
antis = antis[1:]

x = np.vstack((drones,antis))
y = np.vstack((np.ones((len(drones),1)),np.zeros((len(antis),1)))).ravel()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=42)

""" train_test_split function
dataset = np.hstack((x, y))
np.random.shuffle(dataset)
x = dataset[:, :-1]; y = dataset[:, -1]
number_of_test_examples = 20
n = int((64 - number_of_test_examples) / 2)
train_x = np.vstack((x[:n], x[len(x) - n:]))
train_y = np.vstack((np.ones((n,1)),np.zeros((n,1))))
test_x = x[n:len(x) - n]
test_y = np.vstack((np.ones((len(drones) - n, 1)),np.zeros((len(antis) - n, 1))))"""

scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)

model = LogisticRegression(max_iter=5000)
model.fit(train_x,np.train_y)

print((model.score(test_x, test_y)*100).round(2), "percent accuracy")

"""  
# Long Version: Look up for the short version(model.score())
prediction = model.predict(test_x)
accuracy = sum(np.transpose([prediction]) == test_y)
print(accuracy[0], "out of", np.shape(prediction)[0], "examples are predicted correctly")
"""



