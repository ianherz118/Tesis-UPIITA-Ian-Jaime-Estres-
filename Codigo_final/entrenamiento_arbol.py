# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:36:10 2022

@author: ianja
"""

import numpy as np
import pandas as pd
import pickle
import joblib

# Gráficos
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ------------------------------------------------------------------------------
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Configuración warnings
# ------------------------------------------------------------------------------
import warnings
warnings.filterwarnings('once')

dft = pd.read_csv('C:/Users/ianja/OneDrive/Escritorio/Tesis dos/Codigo/ecg.csv')
X_train, X_test, y_train, y_test = train_test_split(
                                        dft.drop(columns = "y"),
                                        dft["y"],
                                        random_state = 123
                                    )
# Creación del modelo
# ------------------------------------------------------------------------------
modelo = DecisionTreeRegressor(
            max_depth         = 3,
            random_state      = 123
          )

# Entrenamiento del modelo
# ------------------------------------------------------------------------------
modelo.fit(X_train.values, y_train)
# filename = 'finalized_model.sav'
# pickle.dump(modelo, open(filename, 'wb'))
# # Estructura del árbol creado
# # ------------------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(12, 5))

# print(f"Profundidad del árbol: {modelo.get_depth()}")
# print(f"Número de nodos terminales: {modelo.get_n_leaves()}")

# plot = plot_tree(
#             decision_tree = modelo,
#             feature_names = dft.drop(columns = "y").columns,
#             class_names   = 'y',
#             filled        = True,
#             impurity      = False,
#             fontsize      = 10,
#             precision     = 2,
#             ax            = ax
#        )

# texto_modelo = export_text(
#                     decision_tree = modelo,
#                     feature_names = list(dft.drop(columns = "y").columns)
#                )
# print(texto_modelo)

# # modelo.save('C:/Users/ianja/OneDrive/Escritorio/Tesis dos/Codigo/autoencoder')

 
# load the model from disk
# loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)

# save
joblib.dump(modelo, "decision_arbol.pkl") 