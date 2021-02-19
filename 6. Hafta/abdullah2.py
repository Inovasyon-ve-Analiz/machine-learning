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

control = int(input("Her zaman fit edilsin mi? EVET icin 1, HYR icin 0 :\n"))

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

test_sizes = np.linspace(0.1,0.99,20)
lst=[]
lst2=[]

scaler = StandardScaler()
scaler.fit(dataX)
dataX = scaler.transform(dataX)

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, 
                                                    test_size=0.1,random_state=3)
xxt = X_test
yyt = y_test

mlp = MLPClassifier(hidden_layer_sizes=(5,5), max_iter=2000)

if not control:
    mlp.fit(X_train, y_train.ravel())

size=X_train.shape[0]

for i in test_sizes:
    xt = X_train[:round(size*i),:]
    yt = y_train[:round(size*i),:]
    if control:
        mlp.fit(xt, yt.ravel())
    
    lst.append(1-mlp.score(xt,yt))
    lst2.append(1-mlp.score(xxt,yyt))
    
a=test_sizes*64

fig, axes = plt.subplots(figsize=(15, 10))
axes.set_title("Learning Curve")
axes.grid()
axes.plot(a,lst2, 'o-', color="g",label="Test error")
axes.plot(a,lst, 'o-', color="r",label="Training error")
axes.legend(loc="best")

plt.show()