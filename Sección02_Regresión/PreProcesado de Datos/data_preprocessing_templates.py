# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:38:06 2023

@author: Ckonny
"""

# Plantilla de Pre Procesado

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
X =     dataset.iloc[:,:-1].values
y =     dataset.iloc[:,3].values

# Dividir el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Escalado de Variable
""" from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """