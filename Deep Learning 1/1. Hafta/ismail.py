<<<<<<< Updated upstream
import cv2
=======
import torchvision
import torch
import pandas

df = pandas.read_csv("weatherAUS.csv")
print(df.axes)
df = df.drop(["MinTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm","RainToday","RainTomorrow"
],axis = 1)
df.to_csv("edited.csv")
temp = ""
locations = []
for i in df["Location"]:
    if not i == temp:
        temp = i
        locations.append(i)
i = 1
for location in locations:
    print(str(i)+") "+location+" "+str(len(df[df["Location"]==location])))
    i+=1
>>>>>>> Stashed changes
