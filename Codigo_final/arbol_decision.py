# -*- coding: utf-8 -*-
"""
Created on Sun May 29 18:56:24 2022

@author: ianja
"""
import joblib
import numpy as np

clf2 = joblib.load("modelo_kmeans.pkl")
a=(1,2,3,4,5,6,7)
b=(1,2,3,4,5,6,7)
ecg_car=[a+b]
ecg_car = np.array(ecg_car)
y=clf2.predict(ecg_car)

print(y)