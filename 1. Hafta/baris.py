import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

df = pd.read_csv('covid_19_data_tr.csv')

confirmed_case = df['Confirmed'].to_numpy().reshape(-1, 1)
day = np.arange(1, len(confirmed_case) + 1).reshape(-1, 1)

dataset = np.concatenate((day, confirmed_case), axis=1)
#print(dataset)

np.random.shuffle(dataset)
x_train, y_train, x_test, y_test = dataset[: int(len(dataset) * 0.75), 0].reshape(-1, 1), \
                                   dataset[: int(len(dataset) * 0.75), 1].reshape(-1, 1), \
                                   dataset[int(len(dataset) * 0.75):, 0].reshape(-1, 1), \
                                   dataset[int(len(dataset) * 0.75):, 1].reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

print('75 gün sonraki vaka sayısı: ' + str(round(regr.predict([[75]])[0, 0])))

fig, axes = plt.subplots()
axes.scatter(day, confirmed_case, c='black')  # dataset
axes.plot(x_test, y_pred, c='red')  # hypothesis
axes.set_title('Day vs Confirmed Case')
axes.set_xlabel('Day'); axes.set_ylabel('Confirmed Case')
plt.show()