import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn.linear_model import LogisticRegression

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
train_x = np.vstack((x[1:26], x[40:]))
train_y = np.vstack((np.ones(25,1), np.zeros(25)))
test_x = x[26:40]
test_y = train_y = np.vstack((np.ones(8,1), np.zeros(6)))

model = LogisticRegression(max_iter=5000)
model.fit(train_x,np.ravel(train_y))

prediction = model.predict(test_x)
print(prediction)   #predicts inversely