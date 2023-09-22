
import matplotlib.pyplot as plt
import seaborn as sns

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

#df = pd.read_csv(r'D:\Data_Analytics\pythonProject1\code\breast-cancer-datasets-and-codes\DB1\DB1\DB1.csv')
df = pd.read_csv(r'D:\Data_Analytics\pythonProject1\code\breast-cancer-datasets-and-codes\DB-csv-files\modDB1.csv')


# basic data preparation
X = np.array(df.drop(['class'], axis=1)) #input
X = X.astype('float32')
y = np.array(df['class'])   #output
# integer encode
y = LabelEncoder().fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state =123)
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)

print("y_train: ")
print(y_train)
print("y_test:")
print(y_test)
#
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)

results_df = pd.DataFrame(models)
#results_df.to_csv('DB1_lazypredict_result.csv')

tmp = models.index
plt.figure(figsize=(20, 5))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="Accuracy",y=models.index, data=models)
plt.xticks(rotation=90)

plt.show()