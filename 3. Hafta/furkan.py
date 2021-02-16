import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import random

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
y = np.vstack((np.ones((len(drones),1)),np.zeros((len(antis),1))))
dataset = np.hstack((x, y))
np.random.shuffle(dataset)
x = dataset[:, :-1]; y = dataset[:, -1]

number_of_test_examples = 20
n = int((64 - number_of_test_examples) / 2)
train_x = np.vstack((x[:n], x[len(x) - n:]))
train_y = np.vstack((np.ones((n,1)),np.zeros((n,1))))
test_x = x[n:len(x) - n]
test_y = np.vstack((np.ones((len(drones) - n, 1)),np.zeros((len(antis) - n, 1))))

scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)

model = LogisticRegression(max_iter=5000)
model.fit(train_x,np.ravel(train_y))

prediction = model.predict(test_x)
accuracy = sum(np.transpose([prediction]) == test_y)
print(accuracy, "out of", np.shape(prediction), "examples are predicted correctly")