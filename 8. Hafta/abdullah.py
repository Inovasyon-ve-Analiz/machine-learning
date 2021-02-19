# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:22:08 2021

@author: yerminal
@website: https://github.com/yerminal
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(10)
s = np.concatenate((np.random.uniform(-200,200,40).reshape(-1,1),np.random.uniform(-200,200,40).reshape(-1,1)),axis=1)

colors = KMeans(n_clusters=6, random_state=3).fit_predict(s)
plt.scatter(s[:, 0], s[:, 1], c=colors)