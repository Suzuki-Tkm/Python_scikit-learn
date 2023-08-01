import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = []
y = []
folder_path = "./final_work/realHuman"
cascade_path = '/Users/apple/project/recognition/haarcascade/opencv/data/haarcascades/haarcascade_frontalface_default.xml'

def get_subfolder_info(folder_path):
    subfolders_info = []
    for e in os.scandir(folder_path):
        subfolders_info.append((e.name, e.path))
    return subfolders_info

subfolder_info = get_subfolder_info(folder_path)
cascade = cv2.CascadeClassifier(cascade_path)

for name, path in subfolder_info:
    for root, d , files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces = cascade.detectMultiScale(image)
                for (x1, y1, w1, h1) in faces:
                    face = image[y1:y1+h1, x1:x1+w1]
                face = cv2.resize(face, (64, 64))
                X.append(face.flatten())
                y.append(name)

X = np.array(X)
y = np.array(y)

# データを学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 学習
svm = Pipeline([('scaler', StandardScaler()), ('linear_svc', LinearSVC(C=17, loss='hinge', max_iter=100000))])
svm.fit(X_train, y_train)

# テスト
y_pred = svm.predict(X_test)

print("正解率:", accuracy_score(y_test, y_pred))
print("適合率(precision)，再現率(recall)，F1スコア，正解率(accuracy)")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print("混同行列")
print(confusion_matrix(y_test, y_pred))