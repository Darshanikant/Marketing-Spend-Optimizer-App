# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:09:50 2024

@author: sunil
"""
# import the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#import data set
data=pd.read_csv(r"C:\Users\sunil\Desktop\DK\vs code\INVERSTMENT PREDICTION\Investment_data.csv")

#dependet vs independent
x=data.iloc[:,:-1]
y=data.iloc[:,4]

#creat a dummy for state
x=pd.get_dummies(x,dtype= int)


# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Save the scaler to a file for later use in the Streamlit app
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)
print("Scaler has been saved as 'scaler.pkl'")

# built regression to fit the train data
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train) 

#prediction
y_pred=lr.predict(x_test)
print("Prediction:- ",y_pred)

# slop
m=lr.coef_
print("slope:- ",m)

#intercept
c=lr.intercept_
print("intercept:- ",c)

#bais
bais=lr.score(x_train,y_train)
print("bais value:- ",bais)

#variance
var=lr.score(x_test,y_test)
print("The variance:- ",var)


# pickel the code
import pickle
filename='inverstment.pkl'
with open(filename,"wb") as file:
    pickle.dump(lr,file)
print('Model has been picked saved in inverstment.pkl ')

# check the path
import os
print(os.getcwd())