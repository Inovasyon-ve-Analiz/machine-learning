import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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

def learningCurve(x, y):
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import ShuffleSplit
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.30, random_state=1)
    
    model = MLPClassifier(hidden_layer_sizes=3, max_iter=5000, random_state=1)
    model.fit(train_x, train_y)
    
    train_sizes = np.linspace(0.1, 1, 5)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    train_sizes, train_scores, validation_scores = learning_curve(model, x, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    
    fig, ax = plt.subplots(2, figsize=(10,15))
    ax[0].plot(train_sizes, 1 - train_scores_mean, label = 'Training Error')
    ax[0].plot(train_sizes, 1 - validation_scores_mean, label = 'Validation Error')
    ax[0].set_xlabel("Training Example"); ax[0].set_ylabel("Error")
    
    ax[1].plot(np.arange(1, model.n_iter_ + 1), model.loss_curve_, label="Training Error")
    ax[1].set_xlabel("Number of Iterations"); ax[1].set_ylabel("Training Error")
    
    ax[0].legend(); ax[1].legend(); 
    plt.subplots_adjust(hspace=0.4)
    plt.show()
    
    
y=y.astype('int')
learningCurve(x, y.ravel())

