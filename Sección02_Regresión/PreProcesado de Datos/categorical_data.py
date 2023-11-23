# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:49:36 2023

@author: Ckonny
"""

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Data.csv')
X =     dataset.iloc[:,:-1].values
y =     dataset.iloc[:,3].values

# Codificar datos Categóricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:,0] = le_X.fit_transform(X[:,0])

#Dummyficar Country
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=float)

# En consola para la visualización de decimales
# np.set_printoptions(suppress=True)

# Dummyficar Purchased
le_y = preprocessing.LabelEncoder()
y = le_y.fit_transform(y)