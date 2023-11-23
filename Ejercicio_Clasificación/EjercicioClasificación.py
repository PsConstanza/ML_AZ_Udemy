# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:29:24 2023

@author: Ckonny
"""

# Ejercicio de Clasificación
# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Limpieza de Texto
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review. split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Crear Bags of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# 1 REGRESIÓN LOGÍSTICA
# Dividir el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Ajustar el modelo de regresión logística al conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train) 

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
                      

# 2 KNN
# Ajustar el clasificador modelo al conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 3 SVM
# Ajustar el SVM modelo al conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 4 KERNEL SVM
# Ajustar el kernel SVM modelo al conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 5 NAIVE BAYES
# Ajustar el Naive Bayes modelo al conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB 
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 6 DECISSION TREE
# Ajustar el clasificador de árbol de decisión  modelo al conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 7 RANDOM FOREST
# Ajustar el clasificador de bosques aleatorios modelo al conjunto de entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 8 CART 
# Ajustar el clasificador CART modelo al conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=2)
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 9 C5.0 
# Ajustar el clasificador C5.0 modelo al conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", min_samples_split=75, max_leaf_nodes=5 )
classifier.fit(X_train, y_train)

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de Confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# 10 MAX ENTROPY
# Ajustar el modelo de regresión logística (MAXEN) al conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, multi_class='multinomial', solver='saga')
classifier.fit(X_train, y_train) 

# Predicción con los resultados del conjunto de testing
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)