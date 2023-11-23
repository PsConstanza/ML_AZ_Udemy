# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 18:58:09 2023

@author: Ckonny
"""

# Regresión Bosques Aleatorios

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Dividir el dataset en conjunto de entrenamiento y testing
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
'''

# Escalado de Variable
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

# Ajustar la Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 300, random_state = 0 )
regression.fit(X, y)


# Predicción de nuestro modelo con Random Forest
y_pred = regression.predict([[6.5]]) #A partir de un valor o un conjunto de testing

# Visualización de los resultados del Random Forest con Grid
X_grid = np.arange(min(X), max(X), 0.01) # para qe sean valores continuos
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión con Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en USD$)")
plt.show()

# Visualización de los resultados del Random Forest
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Reg con Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en USD$)")
plt.show()

