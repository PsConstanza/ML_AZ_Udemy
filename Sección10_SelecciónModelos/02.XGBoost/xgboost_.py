# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 21:22:34 2023

@author: Ckonny
"""

# XG Boost

# PARTE 1: PRE PROCESADO DE DATOS
# Importar Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Variables Categóricas - Dummyficar 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    transformers= [
        ('Geography', OneHotEncoder(drop='first'), [1]),
        ('Gender', OneHotEncoder(drop='first'), [2]),
        ],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=float)
# X = ct.fit_transform(X)

# Dividir el data set en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Aplicar el modelo XGBoostal Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()    
accuracies.std()