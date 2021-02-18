import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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
scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)

scores = []
for i in range(1,7):
    model = MLPClassifier(hidden_layer_sizes=i, random_state=1, max_iter=300)
    model.fit(train_x, train_y)
    scores.append((model.score(test_x, test_y)*100).round(2))

for j in range(1,7):
    print(f"For {j} hidden layer, we get", scores[j - 1], "percent accuracy")