import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


db = pd.read_csv("car_fuel_consumption.csv")
x = np.array(db[["distance", "speed", "temp_outside", "gas_type"]])
x = x[x[:,3] == "E10"][:,:3]; 
y = np.array(db[["consume", "gas_type"]])
y = y[y[:,1] == "E10"][:,0].reshape(-1,1)

x[:, 0] = np.array([float(".".join(i[0].split(","))) for i in x])
y = np.array([float(".".join(i[0].split(","))) for i in y]).reshape(-1,1)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
model = linear_model.LinearRegression()
model.fit(train_x, train_y)

prediction = model.predict(test_x)

accuracy = 100 * (1 - 1/(len(test_y)) * sum(abs(prediction - test_y) / test_y))
print("Model Accuracy:", accuracy[0].round(2), "percent")
