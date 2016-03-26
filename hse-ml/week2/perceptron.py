# IPython log file

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train = pd.read_csv("perceptron-train.csv", names = ['y', 'x1','x2'])
test = pd.read_csv("perceptron-test.csv", names = ['y','x1','x2'])

train_x = train.as_matrix()[:,1:]
test_x = test.as_matrix()[:,1:]
train_y = train.as_matrix()[:,0]
test_y = test.as_matrix()[:,0]

scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

clf = Perceptron()
clf.fit(train_x, train_y)
predictions = clf.predict(test_x)
sc1 = accuracy_score(test_y, predictions)

clf.fit(train_x_scaled, train_y)
predictions = clf.predict(test_x_scaled)
sc2 = accuracy_score(test_y, predictions)

print("{0:.2f}".format(sc2 - sc1))
