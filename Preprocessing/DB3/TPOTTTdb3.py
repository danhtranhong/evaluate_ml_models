# import tpot
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('/Users/minhnguyetnguyen/Documents/AISOUP/AutoML/Database/DB3/std_db3.csv')

#cols = ['age', 'menopause', 'tumor_size', 'inv_nodes','node_caps','breast', 'breast_quad','irradiat', 'Class']
#df[cols] = df[cols].apply(LabelEncoder().fit_transform)
X = np.array(df.drop(['Class'], axis=1))  # input
y = np.array(df['Class'])  # output

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=123)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(generations=10, population_size=50, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot.fit(X_train, y_train)
tpot.export('bestmodelTT.py')
preds = tpot.predict(X_test)
print(accuracy_score(y_test, preds))

