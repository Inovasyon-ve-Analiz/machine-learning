from matplotlib import pyplot as plt
from numpy.random import normal, exponential, uniform
import numpy as np

#normal
mean = 50
std = 0.1
sample = normal(mean,std,10000)

plt.figure(1)
_, bins, _ = plt.hist(sample,300,density=True)
plt.plot(bins,1/(std * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * 0.1**2) ),linewidth=2, color='r')

#exponential
scale = 1
sample = exponential(scale,10000)

plt.figure(2)
_, bins, _ = plt.hist(sample,300,density=True)
plt.plot(bins,np.exp(-bins/scale)/scale,linewidth=2, color='r')

#uniform
low =  0
high = 1
sample = uniform(low,high,10000)

plt.figure(3)
_, bins, _ = plt.hist(sample,300,density=True)
plt.plot(bins,np.ones(bins.shape),linewidth=2, color='r')

plt.show()