"""
Brandon Espina
10/20/2023
COSC_4368
Dr. Lin
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_excel('C:\\Users\Windows\Desktop\COSC_4368\AIPS\PS2\content\\train.xlsx')

train_df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()

embark_dummies = pd.get_dummies(train_df, columns=['Embarked'], dtype=int)
embark_dummies['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

# using average to replace missing numbers.
nan_count = 0
for num in range(len(embark_dummies.Age)):
    if embark_dummies.Age[num] == np.nan:
        continue
    else:
        nan_count += 1

sum_age = embark_dummies.Age.sum()
average = sum_age / nan_count

embark_dummies['Age'].replace(np.nan, round(average), inplace=True)

y = embark_dummies.Survived
separated = embark_dummies[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]

X = separated.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)

print('X_train : ')
print(X_train)

print('')
print('X_test : ')
print(X_test)

print('')
print('y_train : ')
print(y_train)

print('')
print('y_test : ')
print(y_test)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

# ----------KNN-------------
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred_KNN = knn.predict(X_test)

accuracy_KNN = metrics.accuracy_score(y_test, y_pred_KNN)
precision_KNN = metrics.precision_score(y_test, y_pred_KNN)
recall_KNN = metrics.recall_score(y_test, y_pred_KNN)

# ----------MLP--------------
classifier_MLP = MLPClassifier(activation="relu", max_iter=300, solver='adam')
classifier_MLP.fit(X_train, y_train)
y_pred_MLP = classifier_MLP.predict(X_test)

accuracy_MLP = metrics.accuracy_score(y_test, y_pred_MLP)
precision_MLP = metrics.precision_score(y_test, y_pred_MLP)
recall_MLP = metrics.recall_score(y_test, y_pred_MLP)

# ----------SVM--------------
classifier_SVM = svm.SVC(kernel='linear')
classifier_SVM.fit(X_train, y_train)
y_pred_SVM = classifier_SVM.predict(X_test)

accuracy_SVM = metrics.accuracy_score(y_test, y_pred_SVM)
precision_SVM = metrics.precision_score(y_test, y_pred_SVM)
recall_SVM = metrics.recall_score(y_test, y_pred_SVM)

# ----------LogisticRegression--------------
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression().fit(X_train, y_train)
y_pred_LogR = model.predict(X_test)

accuracy_LogR = metrics.accuracy_score(y_test, y_pred_LogR)
precision_LogR = metrics.precision_score(y_test, y_pred_LogR)
recall_LogR = metrics.recall_score(y_test, y_pred_LogR)

print(metrics.mean_squared_error(y_test, y_pred_LogR))

d = {
    'Accuracy': [accuracy_MLP, accuracy_LogR, accuracy_SVM, accuracy_KNN],
    'Model': ['Multi Layer Perceptron', 'Logistic Regression', 'Support Vector Machines', 'KNN'],
    'Precision': [precision_MLP, precision_LogR, precision_SVM, precision_KNN],
    'Recall': [recall_MLP, recall_LogR, recall_SVM, recall_KNN],
}
output = pd.DataFrame(data=d)
print(output)

confusion = metrics.confusion_matrix(y_test, y_pred_LogR)
display = metrics.ConfusionMatrixDisplay(confusion)
display.plot()
plt.title(label="LogR")
plt.show()

confusion = metrics.confusion_matrix(y_test, y_pred_KNN)
display_KNN = metrics.ConfusionMatrixDisplay(confusion)
display_KNN.plot()
plt.title(label="KNN")
plt.show()

confusion = metrics.confusion_matrix(y_test, y_pred_MLP)
display_MLP = metrics.ConfusionMatrixDisplay(confusion)
display_MLP.plot()
plt.title(label="MLP")
plt.show()

confusion = metrics.confusion_matrix(y_test, y_pred_SVM)
display_SVM = metrics.ConfusionMatrixDisplay(confusion)
display_SVM.plot()
plt.title(label="SVM")
plt.show()
