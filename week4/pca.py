# IPython log file

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
data = pd.read_csv('close_prices.csv', index_col=0)
djia = pd.read_csv('djia_index.csv', index_col=0)

pca = PCA(n_components=10)
pca.fit(data)

tr_data = pca.transform(data)

for i in range(1,len(pca.explained_variance_)):
    if sum(pca.explained_variance_ratio_[0:i]) >= 0.9:
        break
print(i)

print("{0:.2f}".format(np.corrcoef(tr_data[:,0],djia['^DJI'].as_matrix())[0,1]))

print(data.columns.values[pca.components_[0].argmax()])
