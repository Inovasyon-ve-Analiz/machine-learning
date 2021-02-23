# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:00:12 2021

@author: yerminal
@website: https://github.com/yerminal
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def convert_comma(val):
    val = str(val)
    new_val = val.replace(',', '.')
    return float(new_val)

df = pd.read_csv('car_fuel_consumption.csv')
df = df[df['gas_type'] == 'E10'][['distance', 'speed', 
                                  'temp_outside', 'consume']].applymap(convert_comma
                                                                       ).to_numpy().reshape(-1, 4)
dataX = df[:,:3]
dataY = df[:,-1]

"""
soe = np.concatenate((dataX[:,0].reshape(-1,1), dataY.reshape(-1,1)), axis=1)
boom = np.array(sorted(soe,key = lambda x: x[0]))
"""

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.25)

model = LinearRegression()

model.fit(X_train,y_train)

distance = 24.7
speed = 58
temp_outside = 12

predict = np.array([distance, speed, temp_outside]).reshape(1, 3)
pred = model.predict(predict)

print("The predicted consume value is %.3f."%pred)

"""
sorr1 = (np.concatenate((X_test[:,0].reshape(-1,1), r_y_pred), axis=1))
sorr2 = (np.concatenate((X_test[:,1].reshape(-1,1), r_y_pred), axis=1))
sorr3 = (np.concatenate((X_test[:,2].reshape(-1,1), r_y_pred), axis=1))

x1,y1 = np.hsplit(np.array(sorted(sorr1,key = lambda x: x[0])),2)
x2,y2 = np.hsplit(np.array(sorted(sorr2,key = lambda x: x[0])),2)
x3,y3 = np.hsplit(np.array(sorted(sorr3,key = lambda x: x[0])),2)

plt.scatter(boom[:,0], boom[:,1], s=10)
plt.plot(x1, y1, color='m')
plt.show()
"""