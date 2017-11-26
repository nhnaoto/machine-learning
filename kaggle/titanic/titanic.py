
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

# 訓練データの読み込み
df = pd.read_csv("./input/train.csv", header=0)

# X : 説明変数列を取り出す
# y : 目的変数列を取り出す
X = df.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']];
y = df.loc[:, ['Survived']]
#print(X.dtypes)

X['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
X.Title.replace(['Mlle', 'Major', 'Lady', 'Sir', 'Jonkheer', 'Countess', 'Capt', 'Mme', 'Don', 'Dona'], ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],inplace=True)
#print(X.Title.value_counts())

# X : one hot encodeing
ohe_columns = ['Sex','Embarked','Title']
X = pd.get_dummies(X, columns=ohe_columns)
X  = X.drop(['Title_other'], axis=1)

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# pipe = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier())])

pipe_knn = Pipeline([('scl',StandardScaler()),('est',KNeighborsClassifier())])
pipe_logistic = Pipeline([('scl',StandardScaler()),('est',LogisticRegression(random_state=1))])
pipe_rf = Pipeline([('scl',StandardScaler()),('est',RandomForestClassifier(random_state=1))])
pipe_gb = Pipeline([('scl',StandardScaler()),('est',GradientBoostingClassifier(random_state=1))])
pipe_mlp = Pipeline([('scl',StandardScaler()),('est',MLPClassifier(hidden_layer_sizes=(100,3), max_iter=500, random_state=1))])
pipe_svm = Pipeline([('scl',StandardScaler()),('est',svm.SVC())])


pipe_names = ['KNN','Logistic','RandomForest','GradientBoosting','MLP','SVM']
pipe_lines = [pipe_knn, pipe_logistic, pipe_rf, pipe_gb, pipe_mlp, pipe_svm]

# 交差検証
from sklearn.model_selection import cross_val_score
for (i,pipe) in enumerate(pipe_lines):
    scores = cross_val_score(pipe, X, y.as_matrix().ravel(), cv=10, scoring='accuracy')
    print('%s: %.3f'%(pipe_names[i],scores.mean()))

# ->0.835



# In[ ]:

# print(SVC().get_params().keys())
# # グリッドサーチ
# from sklearn.model_selection import GridSearchCV
# param_grid_gb = {'est__n_estimators':[50,100],'est__subsample':[0.8, 1.0]}
# param_grid_svm = {'gamma': [0.001, 0.0001], 'kernel': ['rbf']}

# pipes = [pipe_svm, pipe_gb]
# params = [param_grid_svm, param_grid_gb]

# # best_estimator = []
# for elem in zip(pipes, params):
#     print('----------------------------------------------------------------------------------------------')
#     pipe, param = elem[0], elem[1]
#     print('探索空間:%s' % param)
#     gs = GridSearchCV(estimator=pipe, param_grid=param, scoring='accuracy', cv=3)
#     gs = gs.fit(X, y.as_matrix().ravel())
#     best_estimator.append(gs.best_estimator_) 
#     print('Best Score %.6f\n' % gs.best_score_)
#     print('Best Model: %s' % gs.best_estimator_)

# print('探索空間:%s' % param_grid_svm)
# gs = GridSearchCV(estimator=pipe_svm, param_grid=param_grid_svm, scoring='accuracy', cv=3)
# gs = gs.fit(X, y.as_matrix().ravel())
# best_estimator.append(gs.best_estimator_) 
# print('Best Score %.6f\n' % gs.best_score_) 
# print('Best Model: %s' % gs.best_estimator_)



# In[ ]:

# fit 
# pipe_gb_best = Pipeline([('scl',StandardScaler()),('est',GradientBoostingClassifier(random_state=1, subsample=0.8, n_estimators=100))])
# pipe = pipe_gb_best
pipe = pipe_svm
pipe.fit(X, y.as_matrix().ravel())


# In[ ]:

# ここから予測データ
# 予測データの読み込み
df = pd.read_csv("./input/test.csv", header=0)

# A ： 説明変数列を取り出す
A = df.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']];
#print(A.dtypes)

A['Title']=df.Name.str.extract('([A-Za-z]+)\.')
A.Title.replace(['Mlle', 'Major', 'Lady', 'Sir', 'Jonkheer', 'Countess', 'Capt', 'Mme', 'Don', 'Dona'], ['other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other', 'other'],inplace=True)
#print(A.Title.value_counts())

# A : one hot encoding
ohe_columns = ['Sex','Embarked','Title']
A = pd.get_dummies(A, columns=ohe_columns)
A  = A.drop(['Title_other'], axis=1)


# A : の欠損値補完
A = pd.DataFrame(imp.transform(A), columns=A.columns.values)

# A : 予測
pred = pipe.predict(A)

# 予測結果ファイル出力
B = df
B["Survived"] = pred
B = df.loc[:, ['PassengerId', 'Survived']];
B.to_csv("./output/prediction.csv",index=False)

# -> score 0.79904


# In[ ]:



