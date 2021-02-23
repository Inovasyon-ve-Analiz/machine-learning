from numpy import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

x = random.uniform(-200,200,(40,2))

print(x)


model = KMeans(n_clusters=6, random_state=0)
model.fit(x)
colors = {0:"r",1:"b",2:"g",3:"y",4:"purple",5:"black"}

for i in range(len(x)):
    plt.scatter(x[i][0],x[i][1],c=colors[model.labels_[i]])

plt.show()