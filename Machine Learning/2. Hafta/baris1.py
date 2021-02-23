df = pd.read_csv('covid_19_data_tr.csv')

confirmed_case = df['Confirmed'].to_numpy().reshape(-1, 1)
day = np.arange(1, len(confirmed_case) + 1).reshape(-1, 1)

dataset = np.concatenate((day, confirmed_case), axis=1)
#print(dataset)

x_train, x_test, y_train, y_test = train_test_split(day, confirmed_case, test_size=0.25)  # alternative of np.shuffle
"""np.random.shuffle(dataset)
x_train, y_train, x_test, y_test = dataset[: int(len(dataset) * 0.75), 0].reshape(-1, 1), \
                                   dataset[: int(len(dataset) * 0.75), 1].reshape(-1, 1), \
                                   dataset[int(len(dataset) * 0.75):, 0].reshape(-1, 1), \
                                   dataset[int(len(dataset) * 0.75):, 1].reshape(-1, 1)"""

#https://www.sciencedirect.com/science/article/pii/S2468042720300385 --> 3. derece seçtim polinomu
poly_reg = PolynomialFeatures(degree=3)
x_train_poly = poly_reg.fit_transform(x_train)
x_test_poly = poly_reg.fit_transform(x_test)

model = LinearRegression()
model.fit(x_train_poly, y_train)

y_pred = model.predict(x_test_poly)
xtest_val, ypred_val = np.hsplit(np.array(sorted(np.concatenate((x_test, y_pred), axis=1), key=lambda x: x[0])), 2)

print('75 gün sonraki vaka sayısı: ' + str(round(model.predict(poly_reg.fit_transform([[75]]))[0, 0])))

fig, axes = plt.subplots()
axes.scatter(day, confirmed_case, c='black')  # dataset
axes.plot(xtest_val, ypred_val, c='red')  # hypothesis
axes.set_title('Day vs Confirmed Case')
axes.set_xlabel('Day'); axes.set_ylabel('Confirmed Case')
plt.show()