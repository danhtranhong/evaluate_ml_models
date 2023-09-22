import numpy as np
import pandas as pd #import pandas
from tpot import TPOTClassifier
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
  ########not working ###
del()
#database
df = pd.read_csv('/Users/minhnguyetnguyen/Documents/AISOUP/AutoML/Database/DB2/DB2MOD.csv')
cols = ['age', 'menopause', 'tumor_size', 'inv_nodes','node_caps','breast', 'breast_quad','irradiat', 'Class']

df[cols] = df[cols].apply(LabelEncoder().fit_transform)



X = np.array(df.drop(['Class'], axis = 1))

y = np.array(df['Class'])


# Look at the dataset again
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
print(df.head())
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)
tpot = TPOTClassifier(generations=10, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot.fit(X, y)
tpot.export('bestmodelCV.py')