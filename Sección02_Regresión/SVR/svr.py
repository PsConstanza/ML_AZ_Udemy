# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 19:27:14 2023

@author: Ckonny
"""

# SVR

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Ajustar la regresión  con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)

# Predicción de nuestro modelo con SVR
y_pred = sc_y.inverse_transform(regression.predict(np.array(([[6.5]]))).reshape(-1, 1)) #A partir de un valor o un conjunto de testing

# Visualización de los resultados del SVR
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en USD$)")
plt.show()

# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1) # para qe sean valores continuos
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en USD$)")
plt.show()
