import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

def date_to_int(date):
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[8:10])
    n = (year-2007)*365 + day
    for i in range(1,month):
        if i in [1,3,5,7,8,10]:
          n += 31
        elif i in [4,6,9,11]:
          n += 30
        elif i==2:
          n += 28
    if year>2016 or (year == 2016 and month>2):
        n+=3
    elif year>2012 or (year == 2012 and month>2):
        n+=2
    elif year>2008 or (year == 2008 and month>2):
        n+=1
    return n


def get_data(path="weatherAUS.csv",city="Albury",train_size = 0.75):
    df = pd.read_csv(path)

    df = df.drop(["MinTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"
    ],axis = 1)
    df = df.dropna()

    data = {}
    min = 365
    max = 10
    for i in df.iterrows():
        date = date_to_int(i[1]["Date"])
        if date < min:
            min = date
        elif date > max:
            max = date
        try:
            data[date].append((i[1]["Location"],i[1]["MaxTemp"]))
        except KeyError:
            data[date] = [(i[1]["Location"],i[1]["MaxTemp"])]

    x = []
    y = []

    for i in range(min,max):
        try:
            if len(data[i]) == 49 and len(data[i+1]) == 49:
                temp = []
                for j in range(49):
                    temp.append(data[i][j][1])
                    if data[i][j][0] == city:
                        y.append(data[i+1][j][1])
                x.append(temp)
        except KeyError:
            pass
    np_x = np.array(x)
    np_y = np.array(y)
    X_train, X_test, Y_train, Y_test = train_test_split(np_x,np_y,train_size=train_size)
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    path = "weatherAUS.csv"
    city = "Albury"
    get_data(path,city)