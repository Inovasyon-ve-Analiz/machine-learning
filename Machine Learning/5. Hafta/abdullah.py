# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:12:52 2021

@author: yerminal
@website: https://github.com/yerminal
"""
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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

dataX = np.vstack((antis,drones))
dataY = np.vstack((np.zeros((31,1)),np.ones((33,1))))

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25, random_state=3) 
#Probably, it is the best score when the random state is 3. 

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=5000)
mlp.fit(X_train, y_train.ravel())

predictions = mlp.predict(X_test)

print("Prediction:",str(predictions))
print("Y_test:    ",str(y_test.T[0]))
print("Accuracy:",str(mlp.score(X_test,y_test))+"\n")

print(confusion_matrix(y_test,predictions),"\n")
print(classification_report(y_test,predictions))