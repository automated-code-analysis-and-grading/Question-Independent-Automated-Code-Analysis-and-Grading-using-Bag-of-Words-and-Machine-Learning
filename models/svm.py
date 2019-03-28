import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("worksheet1.csv")
# print(data2.head())
X = (data.drop(['Verdict', 'Grade', 'Question', 'Solution','Memory'], axis=1))
Y = (data['Grade'])
# print(X[:10])
# print(Y[:10])
# X['Memory'] = X['Memory'].apply(lambda x: float(str(x)[:-1]))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35)
svclassifier = SVC(kernel='poly', degree=1, C=2)
# svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

y_testList = y_test.tolist()
x_testList = X_test.values
for i in range(len(y_pred)):
	print( y_testList[i], y_pred[i], abs(y_pred[i] - y_testList[i]))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
