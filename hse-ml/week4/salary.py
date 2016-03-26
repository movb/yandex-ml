# IPython log file

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

data = pd.read_csv("salary-train.csv")
data_test = pd.read_csv('salary-test-mini.csv')

data.FullDescription = data.FullDescription.str.lower()
data.FullDescription = data.FullDescription.replace('[^a-zA-Z0-9]',' ', regex=True)

data_test.FullDescription = data_test.FullDescription.str.lower()
data_test.FullDescription = data_test.FullDescription.replace('[^a-zA-Z0-9]',' ', regex=True)

vectorizer = TfidfVectorizer(min_df=5)
descr = vectorizer.fit_transform(data.FullDescription)
test_descr = vectorizer.transform(data_test.FullDescription)

data.LocationNormalized.fillna('nan', inplace=True)
data.ContractTime.fillna('nan', inplace=True)

dict_vectorizer = DictVectorizer()
x_categ = dict_vectorizer.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_categ = dict_vectorizer.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

from scipy.sparse import hstack

x = hstack([descr,x_categ])
x_test = hstack([test_descr,test_categ])

from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.0)
clf.fit(x,data.SalaryNormalized)

result = clf.predict(x_test)

print('{0:.2f} {1:.2f}'.format(result[0], result[1]))
