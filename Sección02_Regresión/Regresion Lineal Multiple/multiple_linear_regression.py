# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:19:13 2023

@author: Ckonny
"""

# Regresión Lineal Múltiple

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('50_Startups.csv')
X =     dataset.iloc[:,:-1].values
y =     dataset.iloc[:,4].values

# Codificar datos Categóricos
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

le_X = preprocessing.LabelEncoder()
X[:,3] = le_X.fit_transform(X[:,3])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)
# En consola para la visualización de decimales
# np.set_printoptions(suppress=True)

# Evitar la trampa de las variables ficticias (eliminamos 1)
X = X[:, 1:]

# Dividir el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Escalado de Variable
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """

# Ajustar el modelo de Regresión Lineal Múltiple  con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predicción de los resultados en el conjunto de Testing
y_pred = regression.predict(X_test)

# Construir el modelo óptimo de RLM utilizando la eliminación hacia atrás
import statsmodels.formula.api as sm
import statsmodels.api as SM
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = SM.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = SM.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = SM.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = SM.OLS(y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = SM.OLS(y, X_opt).fit()
regressor_OLS.summary()