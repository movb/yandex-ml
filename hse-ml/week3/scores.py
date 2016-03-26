# IPython log file

import pandas as pd
import numpy as np
from sklearn import metrics

data = pd.read_csv("classification.csv")

tp = data[(data.true == 1) & (data.pred == 1)].count()[0]
tn = data[(data.true == 0) & (data.pred == 0)].count()[0]
fp = data[(data.true == 0) & (data.pred == 1)].count()[0]
fn = data[(data.true == 1) & (data.pred == 0)].count()[0]

print("{0} {1} {2} {3}".format(tp,fp,fn,tn))

accs = metrics.accuracy_score(data.true, data.pred)
precs = metrics.precision_score(data.true, data.pred)
recs = metrics.recall_score(data.true, data.pred)
f1s = metrics.f1_score(data.true, data.pred)

print("{0:.2f} {1:.2f} {2:.2f} {3:.2f}".format(accs,precs,recs,f1s))

data2 = pd.read_csv("scores.csv")

names = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
scores = []

scores.append(metrics.roc_auc_score(data2.true, data2.score_logreg))
scores.append(metrics.roc_auc_score(data2.true, data2.score_svm))
scores.append(metrics.roc_auc_score(data2.true, data2.score_knn))
scores.append(metrics.roc_auc_score(data2.true, data2.score_tree))

print(names[np.argmax(scores)])

scores = []
prec, rec, thres = metrics.precision_recall_curve(data2.true, data2.score_logreg)
scores.append(np.max(prec[rec >= 0.7]))
prec, rec, thres = metrics.precision_recall_curve(data2.true, data2.score_svm)
scores.append(np.max(prec[rec >= 0.7]))
prec, rec, thres = metrics.precision_recall_curve(data2.true, data2.score_knn)
scores.append(np.max(prec[rec >= 0.7]))
prec, rec, thres = metrics.precision_recall_curve(data2.true, data2.score_tree)
scores.append(np.max(prec[rec >= 0.7]))

print(names[np.argmax(scores)])
