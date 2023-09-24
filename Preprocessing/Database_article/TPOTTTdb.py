# import tpot
import sys
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/minhnguyetnguyen/Documents/AISOUP/AutoML/Database_article/DB8MOD.csv')

#cols = ['age', 'menopause', 'tumor-size', 'inv-nodes','node-caps','breast', 'breast-quad','irradiat', 'class']
#df[cols] = df[cols].apply(LabelEncoder().fit_transform)
X = np.array(df.drop(['class'], axis=1))  # input
y = np.array(df['class'])  # output
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=123)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(generations=10, population_size=50, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot.fit(X_train, y_train)

preds = tpot.predict(X_test)
print(accuracy_score(y_test, preds))

