# IPython log file

import numpy as np
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv("svm-data.csv",names = ['y', 'x1', 'x2'])
x = data.as_matrix()[:,1:]
y = data.as_matrix()[:,0]

clf = SVC(C=100000, random_state=241)
clf.fit(x, y)

print(" ".join(map(str,clf.support_+1)))
