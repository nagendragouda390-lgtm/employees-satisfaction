import pandas as pd
import numpy as np 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score,
                             recall_score, 
                             precision_score, 
                             confusion_matrix)

from functions import *

import joblib

df = pd.read_csv("cleaned_data.csv")

"""
(1470, 32)
"""

object_cols = []
num_cols = []

for col in df.columns:
    if df[col].dtypes == "object":
        object_cols.append(col)
    elif col != "Attrition":
        num_cols.append(col)

X = df.drop("Attrition",axis=1)
y = df["Attrition"]

print(X.columns)
X_train,X_test,y_train,y_test = split(X,y)

models = {"LR":LogisticRegression(),
          "KNN":KNeighborsClassifier(),
          "DTC": DecisionTreeClassifier(),
          "RFC": RandomForestClassifier()}

pipes = []
for key in models.keys():
   pipe = pipe_with_column_transform(StandardScaler(), OneHotEncoder(handle_unknown="ignore"), models[key],num_cols, object_cols,X_train,y_train)
   
   pipes.append(pipe)
   
   acc, recall, precision , conf = metrics(pipe,X_test,y_test)

   #print(f"{key}:\naccuracy : {acc}\nrecall : {recall} \nprecision : {precision} \n conf : {conf}")
   
   
"""

LR:
    accuracy  : 0.860
    recall    : 0.340
    precision : 0.615
    conf      : [[237  10]
                 [ 31  16]]
KNN:
   accuracy  : 0.844
   recall    : 0.149
   precision : 0.538
   conf      : [[241   6]
                [ 40   7]]
DTC:
   accuracy  : 0.762
   recall    : 0.383
   precision : 0.305
   conf      : [[206  41]
                [ 29  18]]
RFC:
   accuracy  : 0.840
   recall    : 0.085
   precision : 0.5
   conf      : [[243   4]
                [ 43   4]]

accurate : LR (86.0)
precise  : LR (61.5)
recall   : DTC(38.3)

"""

scores = validation(pipes[0],X_test,y_test,"recall",3)

df = pd.DataFrame(scores)

"""
no     score
1    : 0.267
2    : 0.500
3    : 0.250
mean : 0.339

"""

joblib.dump(pipes[0],"pipe_ohe.pkl")

print("done!!")




