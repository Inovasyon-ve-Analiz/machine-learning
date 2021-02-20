import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ortalama = float(input('ortalama: '))
varyans = float(input('standart sapma: '))
arr = np.random.normal(loc=ortalama, scale=varyans, size=100)
#print(arr)

sns.displot(arr, kind='kde')
print('hesaplanan ortalama ' + str(arr.mean()) + '\nhesaplanan standart sapma: ' + str(arr.std()))

plt.show()