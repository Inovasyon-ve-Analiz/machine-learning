# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:13:02 2021

@author: yerminal
@website: https://github.com/yerminal
"""

"""
Cikardigim Sonuclar:
    1. Neural classifier, ne kadar karmasik olursa (layer artirmak gibi)
    algoritmanın overfitting karakteri articaktir. Bu sebeple TRAINING ERROR
    sifirlarda gezecektir.
    2. Example sayisi cok az oldugundan da TRAINING ERROR un 0 larda gezme
    ihtimali cok yuksek.(Ancak bu sonuctan pek emin degilim)
"""

import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

control = 1

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

scaler = StandardScaler()
scaler.fit(dataX)
dataX = scaler.transform(dataX)

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, 
                                                    test_size=0.2,random_state=3)
xxt = X_test
yyt = y_test

mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=2000,random_state=5654)

if not control:
    mlp.fit(X_train, y_train.ravel())

size=X_train.shape[0]
test_sizes = np.linspace(0.1,0.99,20)
lst=[]
lst2=[]
for i in test_sizes:
    xt = X_train[:round(size*i),:]
    yt = y_train[:round(size*i),:]
    if control:
        mlp.fit(xt, yt.ravel())
        
    predictionsTest = mlp.predict(xxt)
    predictionsTrain = mlp.predict(xt)
   
    print("\nPredictionTrain:\n",str(predictionsTrain))
    print("PredictionTest:",str(predictionsTest))
    print("Y_test:        ",str(y_test.T[0]))
    print("Accuracy:",str(mlp.score(xxt,yyt))+"\n")
    print("-"*50)
    lst.append(1-mlp.score(xt,yt))
    lst2.append(1-mlp.score(xxt,yyt))
    
a=test_sizes*size

fig, axes = plt.subplots(figsize=(15, 10))
axes.set_title("Learning Curve")
axes.grid()
axes.plot(a,lst2, 'o-', color="g",label="Test error")
axes.plot(a,lst, 'o-', color="r",label="Training error")
axes.legend(loc="best")
plt.show()

