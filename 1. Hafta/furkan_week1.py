import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


db = pd.read_csv("covid_19_data_tr.csv")
x = np.arange(1,42).reshape(-1, 1)
y = np.array(db["Confirmed"]).reshape(-1, 1)

model = linear_model.LinearRegression()
model.fit(x, y)
prediction = int(model.predict(np.array([75]).reshape(1, -1)).ravel())
print("The predicted number of covid-19 case at Day 75:",prediction)

h = model.predict(x)
plt.plot(x, y)
plt.plot(x, h)
plt.show()


