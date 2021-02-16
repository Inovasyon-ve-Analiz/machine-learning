import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn.linear_model import LogisticRegression
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
y = np.vstack((np.ones((len(drones),1)),np.zeros((len(antis),1))))
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.75,shuffle=True)

model = LogisticRegression(max_iter=5000)
model.fit(x_train,np.ravel(y_train))
prediction = model.predict(x_test)

print("Prediction:  "+ str(prediction))
print("Y_test:      "+ str(y_test.T[0]))
print("Accuracy: "+str(model.score(x_test,y_test)))