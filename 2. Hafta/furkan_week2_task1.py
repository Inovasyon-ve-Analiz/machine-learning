import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


db = pd.read_csv("covid_19_data_tr.csv")
x = np.arange(1,42).reshape(-1, 1)
y = np.array(db["Confirmed"]).reshape(-1, 1)

# Ridge() can also be used as linear regression model
degree = 3
model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
model.fit(x,y)
prediction = int(model.predict(np.array([75]).reshape(1, -1)).ravel())
print("The predicted number of covid-19 case at 75th day:",prediction)

X = np.arange(75).reshape(-1, 1)
y_predict = model.predict(X)

plt.figure()
plt.plot(np.arange(75), y_predict ,'b.')
plt.plot(x, y, 'r-')
plt.title("Polynomial regression with degree " + str(degree))
plt.show()