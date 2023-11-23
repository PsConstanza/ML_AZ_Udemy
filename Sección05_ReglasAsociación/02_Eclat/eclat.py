# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:21:46 2023

@author: Ckonny
"""

# Eclat

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
    
# Entrenar el algoritmo Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_length = 2)

# Visualización de los resultados
results = list(rules)

results[0:10]
