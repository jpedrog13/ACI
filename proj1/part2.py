# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import datetime
import skfuzzy as fuzz
from skfuzzy import control

og = pd.read_csv('Proj1_Dataset.csv')

og['Date Time'] = og['Date'] + ' ' + og['Time']
og['Date Time'] = pd.to_datetime(og['Date Time'], format='%d/%m/%Y %H:%M:%S')
og['Time Delta'] = og['Date Time'].diff().dt.total_seconds()
og['Total Secs'] = pd.to_timedelta(og['Time']).dt.total_seconds()

og['CO2 delta'] = og['CO2'].diff() / og['Time Delta']
og['CO2 delta moving window'] = og['CO2'] - (og['CO2'].shift(10))

og = og[~pd.isna(og).any(axis = 1)]
og = og[og['S1Temp'] > 0]
og = og[og['S2Temp'] > 0]
og = og[og['S3Temp'] > 0]

og = og[og['S1Light'] < 1000]
og = og[og['S2Light'] < 1000]
og = og[og['S3Light'] < 1000]

og['Avg S2+S3 Light'] = (og['S2Light'] + og['S3Light'])/2
og['Avg Temp'] = ((og['S1Temp'] + og['S2Temp'] + og['S3Temp'])/3).round(2)
'''
checkCloudy = og[og['Total Secs'] > 32400]
checkCloudy = checkCloudy[checkCloudy['Total Secs'] < 36000]
checkCloudy = checkCloudy[checkCloudy['Persons'] == 0]
checkCloudy = checkCloudy[checkCloudy['S3Light'] > 0]

checkCloudy.groupby('Date', as_index=False)['S3Light'].mean()
'''
og['Binary'] = og['Persons']
og['Binary'] = og['Binary'].where((og['Binary']) > 2, 0)
og['Binary'] = og['Binary'].where((og['Binary']) == 0, 1)

df = og[['CO2 delta moving window', 'Avg Temp', 'Avg S2+S3 Light', 'Binary']]

#og['CO2 delta moving window'] = og['CO2 delta moving window'].where(og['CO2 delta moving window'] > -10, -10)
#og['CO2 delta moving window'] = og['CO2 delta moving window'].where(og['CO2 delta moving window'] < 30, 30)

#og['Avg Temp'] = og['Avg Temp'].where(og['Avg Temp'] > 19, 19)
#og['Avg Temp'] = og['Avg Temp'].where(og['Avg Temp'] < 22, 22)

#og['Avg S2+S3 Light'] = og['Avg S2+S3 Light'].where(og['Avg S2+S3 Light'] < 550, 550)


co2delta = control.Antecedent(np.arange(og['CO2 delta moving window'].min(), og['CO2 delta moving window'].max()+0.01, 0.01), 'CO2 Delta')
temp = control.Antecedent(np.arange(og['Avg Temp'].min(), og['Avg Temp'].max()+0.001, 0.001), 'Avg Temp')
avglight = control.Antecedent(np.arange(og['Avg S2+S3 Light'].min(), og['Avg S2+S3 Light'].max()+1, 0.1), 'Avg S2+S3 Light')
#time = control.Antecedent(np.arange(og['Total Secs'].min(), og['Total Secs'].max()+1, 0.1), 'Dusk/Day/Night')


result = control.Consequent(og['Binary'], 'Result')

co2delta['poor'] = fuzz.trapmf(co2delta.universe, [-200.0, -200.0, -5, 5.0])
co2delta['average'] = fuzz.trapmf(co2delta.universe, [-5.0, 5.0, 15.0, 25.0])
co2delta['good'] = fuzz.trapmf(co2delta.universe, [15.0, 25.0, 300.0, 300.0])

#co2delta['Low'] = fuzz.trimf(co2delta.universe, [-200.0, -40.0, 20])
#co2delta['Medium'] = fuzz.trimf(co2delta.universe, [-40.0, 10, 60])
#co2delta['High'] = fuzz.trimf(co2delta.universe, [10, 60, 300])

#co2delta.automf(3)

#temp['Low'] = fuzz.trapmf(temp.universe, [0, 0, 19.800, 20.2])
#temp['Medium'] = fuzz.trapmf(temp.universe, [19.800, 20.200, 20.800, 21.200])
#temp['High'] = fuzz.trapmf(temp.universe, [20.800, 21.200, 30, 30])

#temp['Low'] = fuzz.trimf(temp.universe, [18, 19.400, 20.600])
#temp['Medium'] = fuzz.trimf(temp.universe, [19.400, 20.700, 21.600])
#temp['High'] = fuzz.trimf(temp.universe, [20.800, 21.600, 30])

temp.automf(3)

#avglight['Low'] = fuzz.trapmf(avglight.universe, [0.0, 0.0, 100.0, 160.0])
#avglight['Medium'] = fuzz.trapmf(avglight.universe, [100.0, 160.0, 300.0, 380.0])
#avglight['High'] = fuzz.trapmf(avglight.universe, [300.0, 380.0, 1000.0, 1000.0])

#avglight['Low'] = fuzz.trimf(avglight.universe, [0.0, 100.0, 200])
#avglight['Medium'] = fuzz.trimf(avglight.universe, [100.0, 230.0, 380.0])
#avglight['High'] = fuzz.trimf(avglight.universe, [230.0, 380.0, 1000.0])

avglight.automf(3)

# time['Dusk'] = fuzz.trapmf(time.universe, [0.0, 0.0, 27000.0, 30600.0])
# time['Day'] = fuzz.trapmf(time.universe, [27000.0, 30600.0, 66600.0, 70200.0])
# time['Night'] = fuzz.trapmf(time.universe, [66600.0, 70200.0, 86399.0, 86399.0])

result['2 or less'] = fuzz.trimf(result.universe, [0, 0, 1])
result['3 or more'] = fuzz.trimf(result.universe, [0, 1, 1])

#result.automf(3)

co2delta.view()
temp.view()
avglight.view()
result.view()

# r1 = control.Rule(co2delta['Low'], result['2 or less'])
# r2 = control.Rule(co2delta['Medium'] & temp['Low'], result['2 or less'])
# r3 = control.Rule(co2delta['Medium'] & temp['Medium'] & (avglight['Low'] | avglight['Medium']), result['2 or less'])
# r4 = control.Rule(time['Day'] & co2delta['Medium'] & temp['Medium'] & avglight['High'], result['3 or more'])
# r5 = control.Rule(time['Day'] & co2delta['Medium'] & temp['High'] & (avglight['High'] | avglight['Medium']), result['3 or more'])
# r6 = control.Rule((time['Night'] | time['Dusk']) & co2delta['Medium'] & temp['Medium'] & avglight['High'], result['2 or less'])
# r7 = control.Rule((time['Night'] | time['Dusk']) & co2delta['Medium'] & temp['High'] & (avglight['High'] | avglight['Medium']),  result['2 or less'])
# r8 = control.Rule(co2delta['Medium'] & temp['High'] & avglight['Low'], result['2 or less'])
# r9 = control.Rule(co2delta['High'] & (temp['Low'] | temp['Medium']) & avglight['Low'], result['2 or less'])
# r10 = control.Rule(time['Day'] & co2delta['High'] & (avglight['Medium'] | avglight['High']), result['3 or more'])
# r11 = control.Rule(time['Day'] & co2delta['High'] & (temp['Low'] | temp['Medium']) & avglight['Medium'], result['3 or more'])
# r12 = control.Rule((time['Night'] | time['Dusk']) & co2delta['High'] & (temp['Low'] | temp['Medium']) & avglight['Medium'], result['2 or less'])
# r13 = control.Rule((time['Night'] | time['Dusk']) & co2delta['High'] & temp['Low']& avglight['High'], result['3 or more'])
# r14 = control.Rule(co2delta['High'] & avglight['High'], result['3 or more'])
# r15 = control.Rule(co2delta['High'] & temp['High'], result['3 or more'])

r1 = control.Rule(co2delta['poor'], result['2 or less'])
r2 = control.Rule(co2delta['average'] & temp['poor'], result['2 or less'])
r3 = control.Rule(co2delta['average'] & temp['average'], result['2 or less'])
r4 = control.Rule(co2delta['average'] & temp['good'] & avglight['poor'], result['2 or less'])
r5 = control.Rule(co2delta['average'] & temp['good'] & (avglight['average'] | avglight['good']), result['3 or more'])
r6 = control.Rule(co2delta['good'] & (temp['average'] | temp['good']), result['3 or more'])
r7 = control.Rule(co2delta['good'] & temp['poor'] & (avglight['poor'] | avglight['average']), result['2 or less'])
r8 = control.Rule(co2delta['good'] & temp['poor'] & avglight['good'], result['3 or more'])


system = control.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8])
sim = control.ControlSystemSimulation(system)


prediction = np.empty([0, 0])
for i in range(10109):
    print(og['CO2 delta moving window'].iloc[i], og['Avg S2+S3 Light'].iloc[i], og['Avg Temp'].iloc[i], og['Total Secs'].iloc[i])
    sim.input['CO2 Delta'] = og['CO2 delta moving window'].iloc[i]
    sim.input['Avg S2+S3 Light'] = og['Avg S2+S3 Light'].iloc[i]
    sim.input['Avg Temp'] = og['Avg Temp'].iloc[i]
    sim.compute()

    prediction = np.append(prediction, sim.output['Result'])
    print('output', i)
    
#Splitting data in training and test sets
training_set, test_set = train_test_split(df, test_size = 0.1, random_state = None, shuffle=True)

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