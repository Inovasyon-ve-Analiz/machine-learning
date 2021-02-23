import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from sklearn.linear_model import LogisticRegression
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
y = np.vstack((np.ones((len(drones),1)),np.zeros((len(antis),1))))

for i in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.75,shuffle=True,random_state=i)
    
    model_nn1 = MLPClassifier(hidden_layer_sizes=(100,),max_iter=5000)
    model_nn2 = MLPClassifier(hidden_layer_sizes=(100,),max_iter=5000)
    model_lr = LogisticRegression(max_iter=5000)
    
  
    model_nn1.fit(x_train,np.ravel(y_train))
    prediction_nn1 = model_nn1.predict(x_test)
    model_nn2.fit(x_train,np.ravel(y_train))
    prediction_nn2 = model_nn2.predict(x_test)
    model_lr.fit(x_train,np.ravel(y_train))
    prediction_lr = model_lr.predict(x_test)

    print("Random State: " + str(i))
    print("Y Test:              " + str(y_test.T[0]))
    print("Prediction LR:       "+ str(prediction_lr))
    print("Prediction NN_1:     "+ str(prediction_nn1))
    print("Prediction NN_2:     "+ str(prediction_nn2))

    print("Accuracy of LR: "+str(model_lr.score(x_test,y_test)))
    print("Accuracy of NN_1: "+str(model_nn1.score(x_test,y_test)))
    print("Accuracy of NN_2: "+str(model_nn2.score(x_test,y_test)))
    print("")
print("Done")