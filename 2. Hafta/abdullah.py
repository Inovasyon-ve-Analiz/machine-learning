# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:13:27 2021

@author: yerminal
@website: https://github.com/yerminal
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

dataY=pd.read_csv('covid_19_data_tr.csv', sep=',',header=None)[1:][2].to_numpy(
    ).reshape(-1,1).astype('float64')
dataX = np.arange(1,dataY.size+1).reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25)

model = PolynomialFeatures(degree=3)
x_poly = model.fit_transform(X_train)

polyReg = LinearRegression()
polyReg.fit(x_poly, y_train)
y_pred = polyReg.predict(model.fit_transform(X_test))
y_pred2 = polyReg.predict(model.fit_transform(np.arange(41,76).reshape(-1,1)))

print("The confirmed cases are %d after 75 days." %polyReg.predict(model.fit_transform([[75]])))

sorr = np.concatenate((np.arange(41,76).reshape(-1,1), y_pred2), axis=1)
x,y = np.hsplit(np.array(sorted(sorr,key = lambda x: x[0])),2)

plt.scatter(dataX[:41], dataY, s=10)
plt.plot(x, y, color='m')
plt.show()
