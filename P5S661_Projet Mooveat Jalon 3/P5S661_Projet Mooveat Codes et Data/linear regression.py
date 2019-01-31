
#Part 1 : importing the necessary models 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn import preprocessing,model_selection,tree,ensemble
import seaborn as sns
#generating and plotting the heatmap
meteo_data = pd.read_csv('data.csv')
meteo_data0 = meteo_data.loc[meteo_data['GRID_NO']==93087]
corr = meteo_data.corr()
print(corr)
sns.heatmap(corr,cmap = 'YlGnBu')
#
meteo_data0.ix[:,'SNOWDEPTH'] = meteo_data0.SNOWDEPTH.fillna(meteo_data0.SNOWDEPTH.median())
train_data = meteo_data0[['Unnamed: 0','GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY'
, 'TEMPERATURE_AVG','WINDSPEED', 'VAPOURPRESSURE', 'PRECIPITATION', 'RADIATION',
 'SNOWDEPTH']]
train_target = meteo_data0[['VALUE']]
print(meteo_data0)
def classify(trainSet, trainLabels, testSet):
    
    lr =  LinearRegression() 
    predictedLabels = np.zeros(testSet.shape[0])  
    lr.fit(trainSet,trainLabels)
    predictedLabels=lr.predict(testSet)
    
    return predictedLabels
#using cross validation to calculate accuracy for linear regression
X = np.array(train_data)
y = np.array(train_target)
# Initialize cross validation
kf = cross_validation.KFold(X.shape[0], n_folds=10)

totalCost = 0 # Variable that will store the correctly predicted intances  

for trainIndex, testIndex in kf:
    trainSet = X[trainIndex]
    testSet = X[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]

    predictedLabels = classify(trainSet, trainLabels, testSet)

    cost = 0
    for i in range(testSet.shape[0]):
        cost += (predictedLabels[i]-testLabels[i])**2
    
        
    print ('cost: ' + str(np.sqrt(float(cost)/(testLabels.size))))
    totalCost += np.sqrt(float(cost)/(testLabels.size))
print ('Total Cost: ' + str(totalCost/10))
print('Total Accuracy:' + str(1-totalCost/10))











