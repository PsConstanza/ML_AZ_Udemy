# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 23:50:12 2023

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

# Tratamiento de los NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3]) 

# Sobreescribimos nuestro imputer con los nuevos datos
X[:,1:3] = imputer.transform(X[:,1:3])