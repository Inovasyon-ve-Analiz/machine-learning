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

model = LogisticRegression(max_iter=5000)
model.fit(x,np.ravel(y))

prediction = model.predict(cv2.resize(cv2.imread("path/to/x_test_image.jpg",0),(a,a)).reshape(1,a**2))
print(prediction)   #predicts inversely