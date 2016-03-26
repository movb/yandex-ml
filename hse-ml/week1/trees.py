import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

data = pd.read_csv('titanic.csv', index_col='PassengerId')

subset = data[['Pclass','Fare','Age','Sex','Survived']].dropna()

conv_sex = preprocessing.LabelEncoder()
subset.Sex = conv_sex.fit_transform(subset.Sex)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(subset[[0,1,2,3]], subset[[4]])

importances = clf.feature_importances_
max_indexes = importances.argsort()[-2:]
labels = ['Pclass','Fare','Age','Sex']

f = open('trees.txt','w')
f.write('{0} {1}'.format(labels[max_indexes[0]],labels[max_indexes[1]]))
f.close()
