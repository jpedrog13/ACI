# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 16:43:13 2022

@author: Pedro
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from cmath import nan
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv('Proj1_Dataset.csv')
original = df
df=df[['S1Temp','S2Temp','S3Temp','S1Light','S2Light','S3Light','CO2','PIR1','PIR2','Persons']]

#Remove outliers
aux = df.iloc[:, 0:7]

df = df[~pd.isna(df).any(axis = 1)]

df = df[df['S1Temp'] > 0]
df = df[df['S2Temp'] > 0]
df = df[df['S3Temp'] > 0]

df = df[df['S1Light'] < 1000]
df = df[df['S2Light'] < 1000]
df = df[df['S3Light'] < 1000]

"""
Q1 = aux.quantile(0.25)
Q3 = aux.quantile(0.75)
IQR = Q3 - Q1
no_outliers = df[~((aux < (Q1 - 10 * IQR)) |(aux > (Q3 + 10 * IQR))).any(axis=1)]
df = no_outliers[~pd.isna(no_outliers).any(axis=1)]
"""

#Split data
training_set, test_set = train_test_split(df, test_size = 0.1, random_state = None, shuffle=True)

#classifying the predictors and target variables as X and Y
x_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
x_test = test_set.iloc[:,0:-1].values
y_test = test_set.iloc[:,-1].values

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(10),max_iter=500,activation = 'relu',solver='adam',random_state=1)
#max_iter=500

kf = KFold(n_splits=4)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
i=0
for train_indices, test_indices in kf.split(x_train):
    classifier.fit(x_train[train_indices], y_train[train_indices])
    print(i, "---", classifier.score(x_train[test_indices], y_train[test_indices]))
    i=i+1

#Predicting y for X_val
y_pred = classifier.predict(x_test)

#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_test)

print(cm)

acc = accuracy_score(y_pred, y_test)

#Printing the accuracy
print("Accuracy of MLPClassifier : ", acc)
      