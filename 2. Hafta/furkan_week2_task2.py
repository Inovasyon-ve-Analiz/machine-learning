import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


db = pd.read_csv("car_fuel_consumption.csv", decimal=",")
db = db[db["gas_type"] == "E10"]
x = np.array(db[["distance", "speed", "temp_outside"]])
y = np.array(db[["consume"]])

""" 
# Long Version: Look up for the short version
db = pd.read_csv("car_fuel_consumption.csv")
x = np.array(db[["distance", "speed", "temp_outside", "gas_type"]])
y = np.array(db[["consume", "gas_type"]])
x = x[x[:,3] == "E10"][:,:3]; 
y = y[y[:,1] == "E10"][:,0].reshape(-1,1)
x[:, 0] = np.array([float(".".join(i[0].split(","))) for i in x])
y = np.array([float(".".join(i[0].split(","))) for i in y]).reshape(-1,1)
"""


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
scaler = preprocessing.StandardScaler().fit(train_x)
train_x = scaler.transform(train_x)
scaler = preprocessing.StandardScaler().fit(test_x)
test_x = scaler.transform(test_x)

model = linear_model.LinearRegression()
model.fit(train_x, train_y)

prediction = model.predict(test_x)
accuracy = 100 * (1 - sum(abs(prediction - test_y) / test_y)/(len(test_y)))
print("Model Accuracy:", accuracy[0].round(2), "percent")

