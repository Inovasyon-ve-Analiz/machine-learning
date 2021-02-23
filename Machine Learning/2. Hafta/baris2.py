import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def convert_comma(val):
    val = str(val)
    new_val = val.replace(',', '.')
    return float(new_val)


df = pd.read_csv('car_fuel_consumption.csv')
df = df[df['gas_type'] == 'E10'][['distance', 'speed', 'temp_outside', 'consume']].applymap(convert_comma).to_numpy().reshape(-1, 4)

x = df[:, :3]
y = df[:, 3]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

distance = float(input("distance değerini giriniz: "))
speed = float(input("speed değerini giriniz: "))
temp_outside = float(input("temp_outside değerini giriniz: "))

predict = np.array([distance, speed, temp_outside]).reshape(-1, 3)
print("tüketilecek yakıt miktarı tahmini: " + str(reg.predict(predict)[0]))