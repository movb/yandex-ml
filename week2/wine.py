# IPython log file

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pd.read_csv("wine.data", names=['Class', 'Alcohol', 'MalicAcid', 'Ash', ' Alcalinity', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280','Proline'])

y = data.Class
x = data.iloc[:,1:]

m = 50
results = np.empty(m)

kf = KFold(y.count(), n_folds=5, shuffle=True, random_state=42)

for i in range(0,m):
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    scores = []
    for train, test in kf:
        neigh.fit(x.iloc[train],y.iloc[train])
        scores.append(neigh.score(x.iloc[test],y.iloc[test]))
        results[i] = np.mean(scores)

print ("{0}, {1:.2f}".format(results.argmax()+1, results.max()))
x_scaled = preprocessing.scale(x)

for i in range(0,m):
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    scores = []
    for train, test in kf:
        neigh.fit(x_scaled[train],y.iloc[train])
        scores.append(neigh.score(x_scaled[test],y.iloc[test]))
        results[i] = np.mean(scores)

print ("{0}, {1:.2f}".format(results.argmax()+1, results.max()))
