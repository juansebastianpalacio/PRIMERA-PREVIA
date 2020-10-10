#!/usr/bin/env python
# coding: utf-8

# In[22]:


# PROCESAMIENTO DIGITAL

# Se importan las librería numpy y las funciones de preprocesamiento
import numpy as np
from sklearn import preprocessing
# Datos de prueba
input_data = np.array([[5.1, -2.9, 3.3, 9.1],
[-1.2, 7.8, -6.1, 4.3],
[3.9, 0.4, 2.1,-2.5],
[7.3, -9.9, -4.5,-6.2],
[2.3, -3.9, 4.5, 3.1]])
print(input_data)


# In[24]:


# Binarizar los datos

data_binarized = preprocessing.Binarizer(threshold=1.6).transform(input_data)
print("\nDatos binarizados:\n", data_binarized)


# In[25]:


# Imprimir la media y la desviación estándar

print("\nANTES:")
print("Media =", input_data.mean(axis=0))
print("Desviación estándar =", input_data.std(axis=0))


# In[26]:


# Remover la media

data_scaled = preprocessing.scale(input_data)
print("\nDESPUÉS:")
print("Media =", data_scaled.mean(axis=0))
print("Desviación estándar =", data_scaled.std(axis=0))


# In[27]:


# Escalamiento Min Max

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0,
1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max escalamiento de datos:\n", data_scaled_minmax)


# In[ ]:




