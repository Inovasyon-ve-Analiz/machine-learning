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
scaler = preprocessing.StandardScaler().fit(test_x)
test_x = scaler.transform(test_x)

def cost_function(x, y, model):
    m = x.shape[0]
    predict = model.predict(x)
    total_cost = -(1 / m) * np.sum(y*np.log() + (1 - y) * np.log(
    1 - hypothesis(theta, x)))
    return total_cost
    
    
for i in range(10,200,10):
    model = MLPClassifier(hidden_layer_sizes=2, max_iter=i)
    model.fit(train_x, train_y)
    print(model.get_params())
    cost_train = 1/(2*len(train_y)) * sum(list(map(lambda x:x**2, model.predict(train_x) - train_y)), 0)
    cost_test = 1/(2*len(test_y)) * sum(list(map(lambda x:x**2, model.predict(test_x) - test_y)), 0)
    print(cost_train)
    
    
    