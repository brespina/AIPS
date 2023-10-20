import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn
import sklearn
from sklearn import metrics
import content

df = pd.read_excel('/Users/laniwang/Desktop/COSC_4368/PS2/content/train.xlsx')
train_df = df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].copy()

embark_dummies = pd.get_dummies(train_df, columns=['Embarked'])
embark_dummies['Sex'].replace(['male','female'],[0,1], inplace=True)

print(embark_dummies.head())
