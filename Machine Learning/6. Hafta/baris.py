# Görev Kodu 1: 5. Haftada gerçekleştirilen training ve test işlemlerinin iteration sayısına bağlı Cost değeri grafiklerini çıkarınız. Bu grafiklerin underfit,overfit durumlarını değerlendiriniz.

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def error_function(m, y_pred, y_output):
    y_pred = y_pred.reshape(-1, 1)
    error = (-1 / m) * sum(np.multiply(y_output, np.log(y_pred + 1e-15)) + np.multiply((1 - y_output), np.log(1 - y_pred + 1e-15)))
    return error


pos_fold = "C:/Users/Sarper/Desktop/software/Machine Learning/Odevler/week-3/drone_pics"
neg_fold = "C:/Users/Sarper/Desktop/software/Machine Learning/Odevler/week-3/other_pics"

ls = list()

for filename in os.listdir(pos_fold):
    img = cv2.imread(os.path.join(pos_fold, filename), 0)
    if img is not None:
        img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
        ls.append(img.ravel())

for filename in os.listdir(neg_fold):
    img = cv2.imread(os.path.join(neg_fold, filename), 0)
    if img is not None:
        img = cv2.resize(img, (300, 300), cv2.INTER_AREA)
        ls.append(img.ravel())

x = np.array(ls)
y = np.array(len(os.listdir(pos_fold)) * [1] + len(os.listdir(neg_fold)) * [0], dtype='float64').reshape(-1, 1)

dataset = np.concatenate((x, y), axis=1)
np.random.shuffle(dataset)
#print(dataset)

x_train, y_train = dataset[: int(len(dataset) / 3), :-1], dataset[: int(len(dataset) / 3), -1].reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

mlp = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000)
# relu --> rectified linear activation, similar to svm unlike sigmoid
# cross entropy --> similar to lr cost function
# adam --> optimization algorithm
mlp.fit(x_train, y_train.ravel())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_test = scaler.transform(x_test)
#print(len(x_train), len(x_test))

error_train = list()
error_test = list()

for i in range(1, len(x_train) + 1):
    x_train_lc, y_train_lc = x_train[:i], y_train[:i].reshape(-1, 1)
    train_pred = mlp.predict(x_train_lc)
    error_train.append(error_function(len(y_train_lc), train_pred, y_train_lc))

for i in range(1, len(x_test) + 1):
    x_test_lc, y_test_lc = x_test[:i], y_test[:i].reshape(-1, 1)
    test_pred = mlp.predict(x_test_lc)
    error_test.append(error_function(len(y_test_lc), test_pred, y_test_lc))

error_train = np.array(error_train)
error_test = np.array(error_test)
#print(error_train, error_test, sep='\n')

fig, axes = plt.subplots()
axes.plot(range(1, len(x_train) + 1), error_train, c='black', label='train set')
axes.plot(range(1, len(x_test) + 1), error_test, c='red', label='test set')
axes.set_title('Learning Curve')
axes.set_xlabel('Train example(m)'); axes.set_ylabel('Error')
axes.legend()
plt.show()