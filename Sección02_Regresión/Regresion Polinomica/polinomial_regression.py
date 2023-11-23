# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:12:48 2023

@author: Ckonny
"""

# Regresión Polinómica

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
X_train = sc_X.fit_transform(x_train)
X_test = sc_X.transform(x_test) """

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) 

# Ajustar con diferents grados
poly_reg = PolynomialFeatures(degree = 11)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) 

# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en USD$)")
plt.show()

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1) # para qe sean valores continuos
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en USD$)")
plt.show()

# Predicción de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))