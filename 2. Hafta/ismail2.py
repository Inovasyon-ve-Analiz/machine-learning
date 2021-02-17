import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


data = pd.read_csv("car_fuel_consumption.csv" ,thousands='.', decimal=',')
data = data[data["gas_type"] =="E10"]
x = np.array(data[["distance", "speed", "temp_outside"]],dtype=np.float64)
consume = data["consume"].astype("float64").to_numpy()


x_train, x_test, y_train, y_test = train_test_split(x,consume,test_size=0.25,shuffle=True)

model = linear_model.LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
score = model.score(x_test,y_test)
print(y_test)
print(prediction)
print(score*100)
