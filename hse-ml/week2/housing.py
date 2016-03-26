# IPython log file

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import cross_validation

data = load_boston()

x = preprocessing.scale(data.data)
y = data.target

results = np.empty((200,2))
i = 0

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
for p in np.linspace(1.0,10.0,num=200):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski',p=p)
    scores=cross_validation.cross_val_score(neigh, x,y, cv=kf, scoring='mean_squared_error')
    results[i,0] = p
    results[i,1] = np.mean(scores)
    i+=1

print(results[results[:,1].argmax(),0])
