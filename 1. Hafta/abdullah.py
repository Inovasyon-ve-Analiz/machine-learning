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

dataY=pd.read_csv('covid_19_data_tr.csv', sep=',',header=None)[1:][2].to_numpy(
    ).reshape(-1,1).astype('float64')
dataX = np.arange(1,dataY.size+1).reshape(-1,1)
dataX_next = np.arange(dataX.size,dataX.size+76).reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(dataX, dataY)
dataY_pred = regr.predict(np.concatenate((dataX, dataX_next)))

print("The confirmed cases are %d after 75 days." %dataY_pred[-1])
plt.scatter(dataX, dataY,  color='black')
plt.plot(np.concatenate((dataX, dataX_next), axis=0), dataY_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()