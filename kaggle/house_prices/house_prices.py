
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

# 訓練データの読み込み
df = pd.read_csv("./input/train.csv", header=0)

# X : 説明変数列を取り出す
# y : 目的変数列を取り出す
X = df.loc[:, ['TotalBsmtSF', 'OverallCond','MSZoning']];
y = df.loc[:, ['SalePrice']]



# X : one hot encodeing
ohe_columns = ['MSZoning']
X = pd.get_dummies(X, columns=ohe_columns)

print(X.dtypes)
print(y.dtypes)

# # X :欠損値補完
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X), columns=X.columns.values)
#print(X.count())

# パイプライン作成 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


pipe_knn = Pipeline([('scl',StandardScaler()),('est',KNeighborsRegressor())])
pipe_ridge = Pipeline([('scl',StandardScaler()),('est',Ridge(random_state=1))])
pipe_rf = Pipeline([('scl',StandardScaler()),('est',RandomForestRegressor(random_state=1))])
pipe_gb = Pipeline([('scl',StandardScaler()),('est',GradientBoostingRegressor(random_state=1))])
#pipe_mlp = Pipeline([('scl',StandardScaler()),('est',MLPRegressor(hidden_layer_sizes=(100,3), max_iter=500, random_state=1))])


pipe_names = ['KNN','ridge','RandomForest','GradientBoosting']
pipe_lines = [pipe_knn, pipe_ridge, pipe_rf, pipe_gb]


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# pipe_ridge.fit(X_train, y_train.as_matrix().ravel())

# print('CV_Test:%.6f' % accuracy_score(y_test.as_matrix().ravel(), pipe_ridge.predict(X_test)))

# 交差検証
from sklearn.model_selection import cross_val_score
for (i,pipe) in enumerate(pipe_lines):
    scores = cross_val_score(pipe, X, y.as_matrix().ravel(), cv=10, scoring='r2')
    print('%s: %.3f'%(pipe_names[i],scores.mean()))

# -> GradientBoosting: 0.492


# In[ ]:

# fit 
# pipe_gb_best = Pipeline([('scl',StandardScaler()),('est',GradientBoostingClassifier(random_state=1, subsample=0.8, n_estimators=100))])
# pipe = pipe_gb_best
pipe = pipe_gb
pipe.fit(X, y.as_matrix().ravel())


# In[ ]:

# ここから予測データ
# 予測データの読み込み
df = pd.read_csv("./input/test.csv", header=0)

# A ： 説明変数列を取り出す
A = df.loc[:, ['TotalBsmtSF', 'OverallCond','MSZoning']];

# A : one hot encoding
ohe_columns = ['MSZoning']
A = pd.get_dummies(A, columns=ohe_columns)

# A : 欠損値補完
A = pd.DataFrame(imp.transform(A), columns=A.columns.values)
print(A.dtypes)
print(A.count())

# A : 予測
pred = pipe.predict(A)

# 予測結果ファイル出力
B = df
B["SalePrice"] = pred
B = df.loc[:, ['Id', 'SalePrice']];
B.to_csv("./output/prediction.csv",index=False)

# -> score RMSE 0.26789


# In[ ]:



