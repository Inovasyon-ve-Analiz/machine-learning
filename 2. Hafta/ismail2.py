import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


data = pd.read_csv("car_fuel_consumption.csv" ,thousands='.', decimal=',')
data = data[data["gas_type"] =="E10"]
distance = data["distance"].astype("float64")
speed = data["speed"].astype("float64")
temp_outside = data["temp_outside"].astype("float64")
consume = data["consume"].astype("float64").to_numpy()

x = np.vstack((distance,speed,temp_outside)).T

x_train, x_test, y_train, y_test = train_test_split(x,consume,test_size=0.75,shuffle=True)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
score = model.score(x_test,y_test)
print(y_test)
print(prediction)
