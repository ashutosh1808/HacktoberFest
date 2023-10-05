#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#fetch california housing dataset
from sklearn.datasets import fetch_california_housing
cf=fetch_california_housing()
data=pd.DataFrame(cf.data,columns=cf.feature_names)
data["Price"]=cf.target
print(data.head())

#features and target
features=data.drop(["Price"],axis=1)
target=data["Price"]

#train and test the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,random_state=0,test_size=0.3)

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

#prediction
med_inc=float(input("enter median income: "))
house_age=float(input("enter median house age: "))
avg_rooms=float(input("enter avg rooms: "))
avg_bd=float(input("enter avg bedrooms: "))
population=float(input("enter population: "))
avg_occ=float(input("enter avg occupancy: "))
latitude=float(input("enter latitude: "))
longitude=float(input("enter longitude: "))
d=[[med_inc,house_age,avg_rooms,avg_bd,population,avg_occ,latitude,longitude]]
res=model.predict(d)[0]
print("$",round(res*100000,0))
