# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 19:53:35 2023

@author: Ckonny
"""

# Redes Neuronales Artificiales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano

# Instalar TensorFlow y Keras
# conda install -c conda-forge keras


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

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# PARTE 2: CONSTRUIR LA RNA

# Importar Keras y Librerías Adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA 
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Añadir la segunda capa oculta
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Añadir la capa de salida
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compilar la RNA
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

# Ajustamos la RNA al conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)


# PARTE 3: EVALUAR MODELO Y CALCULAR PREDICCIONES FINALES

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Elaborar matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)