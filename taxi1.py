# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:53:32 2021

@author: Mohamed
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# training data and testing data
X_train=pd.read_excel('taxi_train.xlsx',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
y_train=pd.read_excel('taxi_train.xlsx',usecols=[13])
X_test=pd.read_excel('taxi_test.xlsx',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
y_test=pd.read_excel('taxi_test.xlsx',usecols=[13])

# modele
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train,y_train) 

# pour visualiser les résultats on choisit 2 colonnes de X_train
x1_train=pd.read_excel('taxi_train.xlsx',usecols=[12])
x2_train=pd.read_excel('taxi_train.xlsx',usecols=[1])

# Visualisation des résultats sur la data d'entrainement

fig = plt.figure(figsize=(7,7))
ax1=plt.axes(projection='3d')
ax1.scatter(x1_train, x2_train, y_train, c='r',s=10)
ax1.scatter(x1_train, x2_train, regressor.predict(X_train), c='b',s=10)
ax1.set_xlabel('X1_train')
ax1.set_ylabel('X2_train')
ax1.set_zlabel('y')
# pour visualiser les résultats on choisit 2 colonnes de X_test
x1_test=pd.read_excel('taxi_test.xlsx',usecols=[12])
x2_test=pd.read_excel('taxi_test.xlsx',usecols=[1])

# Visualisation des résultats sur la data de test
fig = plt.figure(figsize=(7,7))
ax2=plt.axes(projection='3d')
ax2.scatter(x1_test, x2_test, y_test, c='r',s=10)
ax2.scatter(x1_test, x2_test, regressor.predict(X_test), c='b',s=10)
ax2.set_xlabel('X1_test')
ax2.set_ylabel('X2_test')
ax2.set_zlabel('y')

# etude des performances
R1=regressor.score(X_train,y_train)
R2=regressor.score(X_test,y_test)
print('\ntraining: %0.1f' %R1)
print('\ntesting: %0.1f' %R2)




