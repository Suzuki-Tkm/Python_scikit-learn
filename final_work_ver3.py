import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = fetch_olivetti_faces()

person_5 = np.isin(data.target, range(30,35))
X = data.data[person_5]
y = data.target[person_5]

# データを学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 学習
svm = Pipeline([('scaler', StandardScaler()), ('linear_svc', LinearSVC(C=1, loss='hinge', max_iter=100000))])
svm.fit(X_train, y_train)

# テスト
y_pred = svm.predict(X_test)

print("正解率:", accuracy_score(y_test, y_pred))
print("適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print("混同行列")
print(confusion_matrix(y_test, y_pred))