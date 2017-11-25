
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

# 訓練データの読み込み
df = pd.read_csv("./input/train.csv", header=0)

# X : 説明変数列を取り出す
# y : 目的変数列を取り出す
X = df.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']];
y = df.loc[:, ['Survived']]
#print(X.dtypes)

# X : one hot encodeing
ohe_columns = ['Sex']
X = pd.get_dummies(X, columns=ohe_columns)
#print(X.head(5))

# X :欠損値補完
from sklearn.preprocessing import Imputer
#print(X.count())

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X), columns=X.columns.values)

#print(X.count())

# パイプライン作成 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pipe = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier())])

# 交差検証
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(pipe, X, y.as_matrix().ravel(), cv=10, scoring='accuracy')
print(cv_results)

# fit
pipe.fit(X, y.as_matrix().ravel())

# ここから予測データ
# 予測データの読み込み
df = pd.read_csv("./input/test.csv", header=0)

# A ： 説明変数列を取り出す
A = df.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']];
#print(A.dtypes)

# A : one hot encoding
ohe_columns = ['Sex']
A = pd.get_dummies(A, columns=ohe_columns)

# A : の欠損値補完
A = pd.DataFrame(imp.transform(A), columns=A.columns.values)

# A : 予測
pred = pipe.predict(A)

# 予測結果ファイル出力
B = df
B["Survived"] = pred
B = df.loc[:, ['PassengerId', 'Survived']];
B.to_csv("./output/prediction.csv",index=False)

# -> score 0.77511


# In[ ]:



