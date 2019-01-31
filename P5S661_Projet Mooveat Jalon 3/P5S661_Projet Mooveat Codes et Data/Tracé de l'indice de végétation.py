# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 18:00:52 2019

@author: miche
"""
#Ce code permet de tracer l'indice de végétation sur une année donnée sur un GRID donné (25km/25km)

##importation de l'ensemble des bibliothèques nécessaires
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Mise en forme de la data 
meteo_data = pd.read_csv('data.csv') 
print(max(meteo_data['GRID_NO']), min(meteo_data['GRID_NO'])) # max et min des valeurs que peut prendre le numéro de GRID
meteo_data0 = meteo_data.loc[meteo_data['GRID_NO']==93088] #valeur à changer pour modifier le GRID dont on trace l'indice de végétation

## Création de la variable à tracer en abscisse (c'est à dire jour de 0 à 3)
def year(day):
    return day//10000 # les jours sont mis sous la forme 20080301 pour 01/03/2008 

meteo_data1 = meteo_data0
meteo_data1['YEAR'] = meteo_data1['DAY'].apply(year) 
meteo_data1=meteo_data1.loc[meteo_data1['YEAR']==2011] # valeur à changer pour changer l'année du tracé
meteo_data=meteo_data1
meteo_data
#conversion de la date en jours
meteo_data['DAYCONVERT'] = meteo_data0['DAY']%100 
meteo_data['MONTH']= (meteo_data['DAY']-meteo_data['DAY']%100 )//100
meteo_data['MONTH']= ((meteo_data['MONTH']%100)-1)*30
meteo_data['FINALE']= meteo_data['MONTH']+meteo_data['DAYCONVERT']
meteo_data['FINALE']

## Tracé des points bruts non-reliés
plt.figure(1)
axes = plt.gca()
axes.set_xlim([0,365])
axes.set_ylim([0,0.6])
plt.plot(np.array(meteo_data['FINALE']),np.array(meteo_data['VALUE']),'bo')
plt.xlabel('periode en jours')
plt.ylabel('indice de vegetation')

##Tracé de la courbe d'indice de végétation sur une année
plt.figure()
axes = plt.gca()
axes.set_xlim([0,365])
axes.set_ylim([0,1])
plt.plot(np.array(meteo_data['FINALE']),np.array(meteo_data['VALUE']))
plt.xlabel('periode en jours')
plt.ylabel('indice de vegetation')