# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:36:28 2021

@author: yerminal
@website: https://github.com/yerminal
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

control = 1

dataY=pd.read_csv('covid_19_data_tr.csv', sep=',',header=None)[1:][2].to_numpy(
    ).reshape(-1,1).astype('float64')
dataX = np.arange(1,42).reshape(-1,1) # 41 training examples

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, 
                                                    test_size=0.2,random_state=33)#421#80#14#21#24#33
xxt = X_test
yyt = y_test

mlp = linear_model.LinearRegression()

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

    print("PredictionTest:",str(predictionsTest))
    print("\nPredictionTrain:",str(predictionsTrain))
    print("\nY_test:    ",str(y_test.T[0]))
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


