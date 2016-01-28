import numpy as np
import pandas as pd

data = pd.read_csv('titanic.csv', index_col='PassengerId')

# 1. Calculate males and females.

males = data.Sex.value_counts()[0]
females = data.Sex.value_counts()[1]

f = open('output1.txt','w')
f.write('{0} {1}'.format(males, females))
f.close()

# 2. Calculate survived %

survived = float(sum(data.Survived))/data.Survived.count() * 100.0

f = open('output2.txt','w')
f.write("{0:.2f}".format(survived))
f.close()

# 3. First class %

first_class = float(data.Pclass[data.Pclass==1].count())/data.Pclass.count() * 100.0

f = open('output3.txt','w')
f.write("{0:.2f}".format(first_class))
f.close()

# 4. Mean and median of passengers age

pmean = np.mean(data.Age[data.Age.notnull()])
pmedian = np.median(data.Age[data.Age.notnull()])

f = open('output4.txt','w')
f.write("{0:.2f} {1:.0f}".format(pmean, pmedian))
f.close()

# 5. SibSp and Parch Pirson correlation

cor = np.corrcoef(data.SibSp, data.Parch)[0,1]

f = open('output5.txt','w')
f.write("{0:.2f}".format(cor))
f.close()

