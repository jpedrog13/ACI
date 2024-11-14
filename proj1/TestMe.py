# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Read file
filename = sys.argv[1]
df = pd.read_csv(filename)

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

# load the model from disk
classifier = pickle.load(open('finalized_model.sav', 'rb'))

#Creating predictor and target variables
x_test = df.iloc[:,0:-1].values
y_test = df.iloc[:,-1].values

#Predicting for the test set
y_pred = classifier.predict(x_test)

#Computing the confusion matrix
cm = confusion_matrix(y_pred, y_test)

print("\n")
print("Confusion Matrix:")
print(cm)

#Calculating precision, recall and f1
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

