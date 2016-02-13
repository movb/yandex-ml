# IPython log file

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

data = pd.read_csv("data-logistic.csv", header=None)

y = data[0].as_matrix()
x = data[[1,2]].as_matrix()

def gradient_it(x,y,w,k,C):
    w_new = np.empty(len(w))
    l = len(x[:,0])
    for i in range(0,len(w)):
        w_new[i] = w[i] + \
                k*(1.0/l)*np.sum(y*x[:,i]*(1.0 - 1.0/(1.0+np.exp(-y*np.sum(w*x,1))))) - \
                k*C*w[i]
    return w_new


def gradient(x,y,k,C):
    w = np.zeros(len(x[0]))

    for i in range(0,10000):
        w_new = gradient_it(x,y,w,k,C)
        if np.linalg.norm(w-w_new) < 1e-5:
            break
        w = w_new

    print("break after {0} iterations".format(i))

    return w_new

def sigmoid(x,w):
    return 1.0/(1.0 + np.exp(np.sum(-w*x,1)))

non_reg_mark = roc_auc_score(y, sigmoid(x, gradient(x,y,0.1,0)))
reg_mark = roc_auc_score(y, sigmoid(x, gradient(x,y,0.1,10)))

print("{0:.3f} {1:.3f}".format(non_reg_mark, reg_mark))
