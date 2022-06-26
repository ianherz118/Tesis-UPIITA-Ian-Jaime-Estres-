# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:39:17 2022

@author: ianja
"""

# Tratamiento de datos
# ==============================================================================
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')
import joblib

# Preprocesado y modelado
# ==============================================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

X = pd.read_csv('C:/Users/ianja/OneDrive/Escritorio/Tesis dos/Codigo/muestreo_final.csv')

X_scaled = scale(X)
X_scaled = scale(X)
modelo_kmeans = KMeans(n_clusters=7, n_init=25, random_state=123)
modelo_kmeans.fit(X=X_scaled)

# y_predict = modelo_kmeans.predict(X=X_scaled[0:1])
y_predict = modelo_kmeans.predict(X=X_scaled)
joblib.dump(modelo_kmeans, "modelo_kmeans.pkl") 
# clf2 = joblib.load("cluster.pkl")
# ecg_car=[ecg_car]
# ecg_car = np.array(ecg_car)
# y=clf2.predict(ecg_car)

# print(y_predict)
