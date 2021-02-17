from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 

data = pd.read_csv("covid_19_data_tr.csv")
x_train = np.arange(1,42)
x_train = np.append(x_train,47).reshape((-1,1))#hehehehehe
x_predict = np.arange(42,76).reshape((-1,1))

y_train = np.append(data["Confirmed"].values,90980)#hehehehehe

d = 3
model = linear_model.LinearRegression()
model = make_pipeline(PolynomialFeatures(d), linear_model.LinearRegression())
model.fit(x_train, y_train)

y_predict = model.predict(x_predict)
y_train_predict = model.predict(x_train)

print("75. gun vaka sayisi: "+str(y_predict[len(y_predict)-1]))

plt.scatter(x_train,y_train)
plt.plot(x_predict,y_predict)
plt.plot(x_train,y_train_predict)
plt.show()