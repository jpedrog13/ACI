# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Read file
og = pd.read_csv('Proj1_Dataset.csv')
df = og

#Outlier removal / updating
df = df[~pd.isna(df).any(axis = 1)]
df = df[df['S1Temp'] > 0]
df = df[df['S2Temp'] > 0]
df = df[df['S3Temp'] > 0]

df = df[df['S1Light'] < 1000]
df = df[df['S2Light'] < 1000]
df = df[df['S3Light'] < 1000]

df['Date Time'] = df['Date'] + ' ' + df['Time']
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d/%m/%Y %H:%M:%S')
df['Time Delta'] = df['Date Time'].diff().dt.total_seconds()
df['Total Secs'] = pd.to_timedelta(df['Time']).dt.total_seconds()

df=df.drop(df[(df['S1Light']==0) & (df['S2Light']>0) & (df['S3Light']>0) & (df['Total Secs']>28800) & (df['Total Secs']<72000)].index)
df=df.drop(df[(df['S2Light']==0) & (df['S1Light']>0) & (df['S3Light']>0) & (df['Total Secs']>28800) & (df['Total Secs']<72000)].index)
df=df.drop(df[(df['S3Light']==0) & (df['S1Light']>0) & (df['S2Light']>0) & (df['Total Secs']>28800) & (df['Total Secs']<72000)].index)

df = df.iloc[:, 2:12]

#Convert dataset to binary (1 for more than 2 persons and 0 for the rest)
df['Binary'] = df['Persons']
df['Binary'] = df['Binary'].where((df['Binary']) > 2, 0)
df['Binary'] = df['Binary'].where((df['Binary']) == 0, 1)

dfb=df
dfb=dfb[['S1Temp','S2Temp','S3Temp','S1Light','S2Light','S3Light','CO2','PIR1','PIR2','Binary']]
df=df[['S1Temp','S2Temp','S3Temp','S1Light','S2Light','S3Light','CO2','PIR1','PIR2','Persons']]

"""
Binary model
"""

print("training binary model...")

#Splitting data in training and test sets
training_set, test_set = train_test_split(dfb, test_size = 0.25, random_state = None, shuffle=True)

#Creating predictor and target variables
x_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
x_test = test_set.iloc[:,0:-1].values
y_test = test_set.iloc[:,-1].values

#Initializing the MLPClassifier
b_classifier = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=500,activation = 'relu',solver='adam',random_state=1)

#K-fold cross-validation
kf = KFold(n_splits=4)
i=0
for train_indices, test_indices in kf.split(x_train):
    b_classifier.fit(x_train[train_indices], y_train[train_indices])
    print("Fold ", i, "---", b_classifier.score(x_train[test_indices], y_train[test_indices]))
    i=i+1

#Predicting for the test set
y_pred = b_classifier.predict(x_test)

#Computing the confusion matrix
cm = confusion_matrix(y_pred, y_test)

print("\n")
print("Confusion Matrix:")
print(cm)

#Calculating accuracy, precision, recall and f1
acc = accuracy_score(y_pred, y_test)

precision_0 = cm[0][0]/(cm[0][0]+cm[0][1])
recall_0 = cm[0][0]/(cm[0][0]+cm[1][0])
f1_0 = 2*cm[0][0]/(2*cm[0][0]+cm[0][1]+cm[1][0])

precision_1 = cm[1][1]/(cm[1][1]+cm[1][0])
recall_1 = cm[1][1]/(cm[1][1]+cm[0][1])
f1_1 = 2*cm[1][1]/(2*cm[1][1]+cm[1][0]+cm[0][1])

macro_precision = (precision_0+precision_1)/2
macro_recall = (recall_0+recall_1)/2
macro_f1 = (f1_0+f1_1)/2

print("\n")
print("Accuracy of MLPClassifier : ", acc)
print("Precision 0: ", precision_0)
print("Recall 0: ", recall_0)
print("Precision 1: ", precision_1)
print("Recall 1: ", recall_1)
print("Macro-Precision: ", macro_precision)
print("Macro-Recall ", macro_recall)
print("Macro-F1: ", macro_f1)
    
"""
Multi-class model
"""

print("\n")
print("training multi-class model...")
    
#Splitting data in training and test sets
training_set, test_set = train_test_split(df, test_size = 0.1, random_state = None, shuffle=True)

#Creating predictor and target variables
x_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
x_test = test_set.iloc[:,0:-1].values
y_test = test_set.iloc[:,-1].values

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=500,activation = 'relu',solver='adam',random_state=1)

#K-fold cross-validation
kf = KFold(n_splits=4)
i=0
for train_indices, test_indices in kf.split(x_train):
    classifier.fit(x_train[train_indices], y_train[train_indices])
    print("Fold ", i, "---", classifier.score(x_train[test_indices], y_train[test_indices]))
    i=i+1

#Predicting for the test set
y_pred = classifier.predict(x_test)

#Computing the confusion matrix
cm = confusion_matrix(y_pred, y_test)

print("\n")
print("Confusion Matrix:")
print(cm)

#Calculating accuracy, precision, recall and f1
acc = accuracy_score(y_pred, y_test)

precision_0 = cm[0][0]/(cm[0][0]+cm[0][1]+cm[0][2]+cm[0][3])
recall_0 = cm[0][0]/(cm[0][0]+cm[1][0]+cm[2][0]+cm[3][0])
f1_0 = 2*cm[0][0]/(2*cm[0][0]+cm[0][1]+cm[0][2]+cm[0][3]+cm[1][0]+cm[2][0]+cm[3][0])

precision_1 = cm[1][1]/(cm[1][0]+cm[1][1]+cm[1][2]+cm[1][3])
recall_1 = cm[1][1]/(cm[0][1]+cm[1][1]+cm[2][1]+cm[3][1])
f1_1 = 2*cm[1][1]/(2*cm[1][1]+cm[1][0]+cm[1][2]+cm[1][3]+cm[0][1]+cm[2][1]+cm[3][1])

precision_2 = cm[2][2]/(cm[2][0]+cm[2][1]+cm[2][2]+cm[2][3])
recall_2 = cm[2][2]/(cm[0][2]+cm[1][2]+cm[2][2]+cm[3][2])
f1_2 = 2*cm[2][2]/(2*cm[2][2]+cm[2][0]+cm[2][1]+cm[2][3]+cm[0][2]+cm[1][2]+cm[3][2])

precision_3 = cm[3][3]/(cm[3][0]+cm[3][1]+cm[3][2]+cm[3][3])
recall_3 = cm[3][3]/(cm[0][3]+cm[1][3]+cm[2][3]+cm[3][3])
f1_3 = 2*cm[3][3]/(2*cm[3][3]+cm[3][0]+cm[3][1]+cm[3][2]+cm[0][3]+cm[1][3]+cm[2][3])

macro_precision = (precision_0+precision_1+precision_2+precision_3)/4
macro_recall = (recall_0+recall_1+recall_2+recall_3)/4
macro_f1 = (f1_0+f1_1+f1_2+f1_3)/4

print("\n")
print("Accuracy of MLPClassifier : ", acc)
print("Precision 0: ", precision_0)
print("Recall 0: ", recall_0)
print("Precision 1: ", precision_1)
print("Recall 1: ", recall_1)
print("Precision 2: ", precision_2)
print("Recall 2: ", recall_2)
print("Precision 3: ", precision_3)
print("Recall 3: ", recall_3)
print("Macro-Precision: ", macro_precision)
print("Macro-Recall ", macro_recall)
print("Macro-F1: ", macro_f1)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'));
 


