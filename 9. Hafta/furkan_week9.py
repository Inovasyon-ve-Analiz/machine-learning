import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 0.5, 0.1
x = np.random.normal(mu, sigma, 100)
mean_x = np.mean(x)
std_x = np.std(x)
print(mean_x, std_x)