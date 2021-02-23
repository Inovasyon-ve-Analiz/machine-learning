#https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

coordinates = np.random.uniform(-200, 200, (40, 2))
#print(coordinates)

# by looking elbow graph, it is better to set cluster number 3-4 instead of 6
km = KMeans(
    n_clusters=6, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)

y_km = km.fit_predict(coordinates)
label_code = np.unique(y_km)
#print(y_km)

fig, axes = plt.subplots()
x_cord = coordinates[:, 0]
y_cord = coordinates[:, 1]

#print(x_cord, y_cord, sep='\n')

colors = {0: 'red', 1: 'yellow', 2: 'cyan', 3: 'magenta', 4: 'orange', 5: 'purple', 6: 'pink', 7: 'green'}  # bunch of colors

for i in range(len(x_cord)):
    axes.scatter(x_cord[i], y_cord[i], c=colors[y_km[i]])
    if i < len(label_code):
        axes.scatter(km.cluster_centers_[i, 0], km.cluster_centers_[i, 1], c=colors[len(label_code) - 1 - i], s=150, marker='*', edgecolor='black')

plt.show()

"""
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(coordinates)
    distortions.append(km.inertia_)

# plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')

plt.show()
"""