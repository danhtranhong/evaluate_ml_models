import numpy as np
import pydot
from numpy import asarray
import pandas as pd #import pandas
from lazypredict.Supervised import LazyClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

#df = pd.read_csv('G:/new researches/breast cancer text paper/databases/DB2/DB2MOD.csv')
#df = pd.read_csv(r'D:\Data_Analytics\pythonProject1\code\breast-cancer-datasets-and-codes\DB2\DATABASE\DB2MOD2.csv')


df = pd.read_csv(r'D:\Data_Analytics\pythonProject1\code\breast-cancer-datasets-and-codes\DB2\DATABASE\DB2MOD2.csv')


# basic data preparation
X = np.array(df.drop(['class'], axis=1)) #input
X = X.astype('float32')
y = np.array(df['class'])   #output
# integer encode
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)


results_df = pd.DataFrame(models)
results_df.to_csv('DB2_lazypredict_result.csv')