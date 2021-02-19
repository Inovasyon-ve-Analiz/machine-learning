import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


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

model = MLPClassifier(hidden_layer_sizes=2, max_iter=200)
model.fit(train_x, train_y)
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print(cv)
train_sizes = np.linspace(0.1, 1, 10)
train_sizes, train_scores, validation_scores = learning_curve(MLPClassifier(), x, y, cv=cv, n_jobs=12,  train_sizes=train_sizes)
print(train_sizes)
train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis = 1)

plt.plot(train_sizes, 1 - train_scores_mean, label = 'Training error')
plt.plot(train_sizes, 1 - validation_scores_mean, label = 'Validation error')
plt.legend()
#plt.ylim(0,1)
plt.show()