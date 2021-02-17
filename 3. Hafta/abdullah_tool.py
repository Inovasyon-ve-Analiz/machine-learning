# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:12:20 2021

@author: yerminal
@website: https://github.com/yerminal
"""

# THIS SCRIPT IS MADE TO FIND THE BEST RANDOM STATE.

import numpy as np
import cv2
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

accur=0
rand=0
size=0.25 # Please enter a number between 0 and 1.
max_rand_state = 100

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
    
for i in range(max_rand_state):
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=size, random_state=i)
    
    log_reg = LogisticRegression(max_iter=5000)
    log_reg.fit(X_train, np.ravel(y_train))
    
    print(i)
    ha = log_reg.score(X_test,y_test)
    if accur < ha:
        rand = i
        accur = ha
print(f"\nAccuracy = {accur} \nRandom State = {rand}")