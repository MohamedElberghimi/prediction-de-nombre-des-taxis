# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:13:08 2021

@author: Mohamed
"""

import statistics 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

# training data
# chaque element désigne l'ecart-type de la colonne correspondante
ecart_train=[1817.23,1781.14,1618.64,1526.69,1302.10,1060.40,1261.73,1380.87,1338.17,1319.23,1413.29,1923.71,6.20]
# chaque element désigne la moyenne de la colonne correspondante
mean_train=[2419.45,2257.24,1982.71,2243.73,1744.17,1680.69,1763.89,2055.48,2880.34,2687.59,2825.47,4216.77,12.49]

x1=pd.read_excel('taxi_train.xlsx',usecols=[0])
mean1=np.ones(x1.shape)*mean_train[0]
X_train=(x1-mean1)/ecart_train[0]

for i in range(1,13):
    x=pd.read_excel('taxi_train.xlsx',usecols=[i])
    meani=np.ones(x.shape)*mean_train[i]
    x=(x-meani)/ecart_train[i]
    X_train = np.hstack((X_train,x))

y_train=pd.read_excel('taxi_train.xlsx',usecols=[13])

# testing data
# chaque element désigne l'ecart-type de la colonne correspondante
ecart_test=[1813.90,1777.37,1617.16,1528.30,1301.55,1049.87,1256.40,1388.57,1337.82,1313.50,1408.23,1935.41,6.10]
# chaque element désigne la moyenne de la colonne correspondante
mean_test=[2417.66,2255.50,1978.08,2238.03,1737.55,1680.20,1761.33,2044.77,2876.65,2686.31,2824.10,4209.23,12.62]

x2=pd.read_excel('taxi_test.xlsx',usecols=[0])
mean2=np.ones(x2.shape)*mean_test[0]
X_test=(x2-mean2)/ecart_test[0]

for i in range(1,13):
    x=pd.read_excel('taxi_test.xlsx',usecols=[i])
    meani=np.ones(x.shape)*mean_test[i]
    x=(x-meani)/ecart_test[i]
    X_test = np.hstack((X_test,x))

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