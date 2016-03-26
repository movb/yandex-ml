# IPython log file

import numpy as np
from sklearn import datasets
from sklearn import grid_search
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

grid = {'C': np.power(10.0, np.arange(-5,6))}
cv = KFold(newsgroups.target.size, n_folds=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(x, y)

best_c = 0
max_res = 0
for a in gs.grid_scores_:
    if a.mean_validation_score > max_res:
        max_res = a.mean_validation_score
        best_c = a.parameters['C']

clf = svm.SVC(kernel='linear', random_state=241, C=best_c)
clf.fit(x,y)

max_elements = np.argpartition(np.abs(clf.coef_.toarray()[0]), -10)[-10:]
max_words = [vectorizer.get_feature_names()[elem] for elem in max_elements]

print(" ".join(sorted(max_words)))
