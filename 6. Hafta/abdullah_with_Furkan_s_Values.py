# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:27:00 2021

@author: yerminal
@website: https://github.com/yerminal
"""
# !! ATTENTION !!
# This curve looks like good only if I use Furkan's values. 
# (hidden_layer_sizes=3, MLP -> random_state=1, ShuffleSplit -> random_state=0 
# If not, it is very rare to get that graph.

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import cv2
import glob
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, (1-train_scores_mean) - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, (1-test_scores_mean) - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, (1-train_scores_mean), 'o-', color="r",
                 label="Training error")
    axes.plot(train_sizes, 1-test_scores_mean, 'o-', color="g",
                 label="Cross-validation error")
    axes.legend(loc="best")

    return plt

a = 100
drones = np.ndarray((1,a**2))

for filename in glob.glob('../dataset/drone/*.jpg'):

    img = cv2.resize(cv2.imread(filename,0),(a,a)).reshape(1,a**2)[0]
    drones = np.vstack((drones,img))
   
drones = drones[1:]

antis = np.ndarray((1,a**2))

for filename in glob.glob('../dataset/anti_drone/*.jpg'):

    img = cv2.resize(cv2.imread(filename,0),(a,a)).reshape(1,a**2)[0]
    antis = np.vstack((antis,img))

antis = antis[1:]

dataX = np.vstack((antis,drones))
dataY = np.vstack((np.zeros((31,1)),np.ones((33,1)))) 

scaler = StandardScaler()
scaler.fit(dataX)

dataX = scaler.transform(dataX)

fig, axes = plt.subplots(figsize=(15, 15))

title = "Learning Curve"

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

estimator = MLPClassifier(hidden_layer_sizes=(3), max_iter=2000,random_state=1) 
plot_learning_curve(estimator, title, dataX, dataY.ravel(), axes, ylim=(0, 1.1),
                    cv=cv, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5))

plt.show()