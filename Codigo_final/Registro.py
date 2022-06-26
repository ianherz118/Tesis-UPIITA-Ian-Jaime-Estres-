# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:26:29 2022

@author: ianja
"""
import serial
import numpy as np 
from scipy.signal import butter, lfilter
import neurokit2 as nk
import pandas as pd
import pymysql as my
from tensorflow import keras
import joblib


model1 = keras.models.load_model('save_autoencoder/')
deArduino = serial.Serial("COM3",9600)

m_ECG=[]
deltha=[]
low_alpha=[]
high_alpha=[]

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def car_EEG(signal):
    signal = np.array(signal)
    a=nk.entropy_sample(signal)
    b=nk.entropy_approximate(signal)
    c=nk.entropy_fuzzy(signal)
    d=nk.fractal_correlation(signal)
    e=nk.entropy_shannon(signal)
    f=nk.fractal_dfa(signal, overlap=False, rms=True)
    return a,b,c,d,e,f

    return
    
conta=0
try:
    while(conta<30000):
       
       while(deArduino.inWaiting()==0):
         pass
       datoString = deArduino.readline()
       a = datoString.splitlines()
       
       b=str(a[0])
       c=b.replace("b","")
       d=c.replace("'","")
       if(len(d)>4):
           d=d.split(',')
           deltha.append(float(d[3]))
           low_alpha.append(float(d[5]))
           high_alpha.append(float(d[6]))
           
       else: 
           d=int(d)
           m_ECG.append(d)
           conta+=1
       print(conta)
        
    deArduino.close()

    deltha=(np.fft.ifft(deltha)).real
    low_alpha=(np.fft.ifft(low_alpha)).real
    high_alpha=(np.fft.ifft(high_alpha)).real
    
    m_ECG = m_ECG-np.mean(m_ECG)
    
    deltha=deltha-np.mean(deltha)
    low_alpha=low_alpha-np.mean(low_alpha)
    high_alpha=high_alpha-np.mean(high_alpha)
    
    m_ECG= butter_lowpass_filter(m_ECG, 3.667 , 30 , 6)
    deltha= butter_lowpass_filter(deltha, 3.667 , 30 , 6)
    low_alpha= butter_lowpass_filter(low_alpha, 3.667 , 30 , 6)
    high_alpha= butter_lowpass_filter(high_alpha, 3.667 , 30 , 6)
    
    try:
        data = pd.DataFrame({"ECG" : m_ECG}) 
        data = data.squeeze()
        processed_data, info = nk.bio_process(data, sampling_rate=200)
        results = nk.bio_analyze(processed_data, sampling_rate=200)
        ecg_car1=[results['ECG_Rate_Mean'][0] , results['HRV_MeanNN'][0], results['HRV_PAS'][0], results['HRV_MCVNN'][0],results['HRV_SampEn'][0], results['HRV_MFDFA_alpha1_Width'][0]]
        clf3 = joblib.load("decision_arbol.pkl")
        # ecg_car=[1.23,0.8,0.65,0.54,0.55,0.78]
        ecg_car=[ecg_car1]
        ecg_car = np.array(ecg_car)
        y=clf3.predict(ecg_car)     
        ecg_car1.append(int(y[0]))
        # ecg_car_sql=[results['ECG_Rate_Mean'][0] , results['HRV_MeanNN'][0], results['HRV_LF'][0], results['HRV_PAS'][0], results['HRV_MCVNN'][0],results['HRV_SampEn'][0], results['HRV_MFDFA_alpha1_Width'][0],y[0]]
        print(y)
        
    except Exception:
        print("Error procesar datos señal ECG, vuelva a realizar el procedimiento ")
        pass
    
    try:
        a0,b0,c0,d0,e0,f0=car_EEG(deltha)
        eeg_carac1=[a0[0],b0[0],c0[0],d0[0],e0[0],f0[0]]
        # eeg_carac=[1.23,0.8,0.65,0.54,0.55,0.78]
        eeg_carac=[eeg_carac1]
        eeg_carac = np.array(eeg_carac)
        X_pred = model1.predict(eeg_carac)
        ecm = np.mean((eeg_carac-X_pred)**2)
        umbral_fijo = 0.065
        y =1 if ecm > umbral_fijo else 0; 
        print(X_pred.shape)
        print(y)
        eeg_carac1.append(y)
    except Exception:
        print("Error procesar datos señal EEG, vuelva a realizar el procedimiento ")
        pass
    

except Exception:
     print("Error en el proceso de registro de la señal")
     pass    
 
clf2 = joblib.load("modelo_kmeans.pkl")
# a=(1,2,3,4,5,6,7)
# b=(1,2,3,4,5,6,7)
# ecg_car=[a+b]
ecg_car=[eeg_carac1+ecg_car1]
ecg_car = np.array(ecg_car)
y=clf2.predict(ecg_car)
print(y)
eeg_carac1.append(int(y))
ecg_car1.append(int(y))

try:
    connection= my.connect(
        host="localhost",
        user="root",
        password="118Schwester118",
        db="registro_EEG_ECG")
    
    cursor= connection.cursor()
    sql="INSERT INTO EEG(Sample_D,Aprox_D,Fuzzy_D,Corre_D,Shannon_D,DFA_D,y,cluster) VALUES(%s, %s, %s, %s,%s, %s, %s, %s)"
    cursor.execute(sql,eeg_carac1)
    sql="INSERT INTO ECG(HR,MEDIAN_RR,LF,MEDIAN_REL_RR,SAMPEN,PNN50,y,cluster) VALUES(%s, %s, %s, %s,%s, %s, %s, %s)"
    cursor.execute(sql,ecg_car1)
    connection.commit()
    print("Registro exitoso SQL daatabase")

except my.Error as error:
    print("Failed to insert into MySQL table {}".format(error))
    connection.commit()

        
