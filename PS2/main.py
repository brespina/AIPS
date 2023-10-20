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
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



df = pd.read_excel('C:\\Users\Windows\Desktop\COSC_4368\AIPS\PS2\content\\train.xlsx')


train_df = df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].copy()

embark_dummies = pd.get_dummies(train_df, columns=['Embarked'])
embark_dummies['Sex'].replace(['male','female'],[0,1], inplace=True)


# using average to replace missing numbers.
nan_count = 0
for num in range(len(embark_dummies.Age)):
  if embark_dummies.Age[num] == np.nan:
    continue
  else:
    nan_count += 1


sum = embark_dummies.Age.sum()
average = sum / nan_count


embark_dummies['Age'].replace(np.nan, round(average), inplace=True)


y = embark_dummies.Survived
separated = embark_dummies[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
# print(y)
# print(separated)

X = separated.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=71)

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
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

model = LogisticRegression().fit(X_train,y_train)
y_pred_LogR = model.predict(X_test)
print(y_pred_LogR)
print(metrics.mean_squared_error(y_test,y_pred_LogR))

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_LogR))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

y_pred_KNN = knn.predict(X_test)
print(y_pred_KNN)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_KNN))
classifier_MLP = MLPClassifier(hidden_layer_sizes=(150, 100, 50), activation="relu", max_iter=300, solver='adam')
classifier_MLP.fit(X_train, y_train)

y_pred_MLP = classifier_MLP.predict(X_test)
print(y_pred_MLP)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred_MLP))
classifier_SVM = svm.SVC(kernel='linear')
classifier_SVM.fit(X_train, y_train)


y_pred_SVM = classifier_SVM.predict(X_test)
print(y_pred_SVM)
print('')
print("LogR: ")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_LogR))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred_LogR))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred_LogR))

print('')
print("KNN: ")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_KNN))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred_KNN))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred_KNN))

print('')
print("MLP: ")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_MLP))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred_MLP))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred_MLP))

print('')
print("SVM: ")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred_SVM))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred_SVM))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred_SVM))
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