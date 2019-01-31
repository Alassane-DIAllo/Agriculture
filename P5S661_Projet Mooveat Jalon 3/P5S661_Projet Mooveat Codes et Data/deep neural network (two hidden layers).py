# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:40:53 2019

@author: othmane jebbari
"""
#part 1 : importing the necessary models 
from keras.callbacks import ModelCheckpoint
from sklearn import cross_validation
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
import seaborn as sns
from sklearn import preprocessing
from keras.layers import Dropout
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#Part 2 :getting the data 
meteo_data = pd.read_csv('data.csv')
meteo_data['SNOWDEPTH'] = meteo_data.SNOWDEPTH.fillna(meteo_data.SNOWDEPTH.median())
meteo_data=meteo_data.sort_values(['DAY','GRID_NO'])
def year(day):
    return day//10000
meteo_data1 = meteo_data
meteo_data1['YEAR'] = meteo_data1['DAY']
meteo_data1['YEAR'] = meteo_data1['DAY'].apply(year)
train_target=meteo_data1.loc[meteo_data1['YEAR']<2015].ix[:,'VALUE']
test_target =meteo_data1.loc[meteo_data1['YEAR']==2015].ix[:,'VALUE']
meteo_data1=meteo_data1[['Unnamed: 0','GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY'
, 'TEMPERATURE_AVG','WINDSPEED', 'VAPOURPRESSURE', 'PRECIPITATION', 'RADIATION',
 'SNOWDEPTH','YEAR']]

#Part 3 : separating the trainset and the testset
train = meteo_data1.loc[meteo_data1['YEAR']<2015]
test = meteo_data1.loc[meteo_data1['YEAR']==2015]

#Part 4 : scaling the training and the test data
x = train.values
min_max_scaler = preprocessing.MinMaxScaler()
x=min_max_scaler.fit_transform(train)
train = pd.DataFrame(x)

y = test.values 
min_max_scaler = preprocessing.MinMaxScaler()
y=min_max_scaler.fit_transform(test)
test = pd.DataFrame(y)

#Part 5 : Build the neural network 
NN_model = Sequential()

#Part 6 : add The Input Layer :
NN_model.add(Dense(train.shape[1], kernel_initializer='random_uniform',input_dim = train.shape[1], activation='relu'))

#Part 7 : Use Dropout to avoid overfitting 
NN_model.add(Dropout(0.2, input_shape=(train.shape[1],)))

#Part 8 : add The Hidden Layers :
#NN_model.add(Dense(train.shape[1]//2, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(train.shape[1]//4, kernel_initializer='normal',activation='relu'))

#Part 9 : add The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

#Part 10 : Compile the network :
NN_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
NN_model.summary()

#Part 11 : Define a checkpoint callback
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

#Part 12 : Train the neural network
NN_model.fit(train, train_target, epochs=5, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

#Part 13 : Test the model and calculate accuracy
predictedLabels = NN_model.predict(test)
cost = 0
for i in range(test_target.shape[0]):
    cost += (float(predictedLabels[i])-test_target.values[i])**2
cost = np.sqrt(float(cost)/(test_target.size))
print ('cost: ' + str(cost))
accuracy = 1-cost
print(accuracy)

#Part 14 : Plot the real values and the predicted values for GRID_NO = 93087
test['VALUE']=pd.DataFrame(test_target.values)
test['PREDICTEDVALUE']=pd.DataFrame(predictedLabels)
def year(day):
    return day//10000
test_one_grid=test.loc[meteo_data1['GRID_NO']==93087]
plt.plot(np.array(test_one_grid['DAY']),np.array(test_one_grid['VALUE']))
plt.plot(np.array(test_one_grid['DAY']),np.array(test_one_grid['PREDICTEDVALUE']))












