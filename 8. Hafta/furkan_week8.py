import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


x = np.random.randint(low=-200, high=200, size=(40,2))
model = KMeans(n_clusters=6, random_state=0).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=model)