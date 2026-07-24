import pandas as pd
import numpy as np 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

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

object_cols = []

for col in df.columns:
    if df[col].dtypes == "object":
        object_cols.append(col)

X = df.drop("Attrition",axis=1)
y = df["Attrition"]

X_train,X_test,y_train,y_test = split(X,y)

oe = OrdinalEncoder(handle_unknown="use_encoded_value",
                    unknown_value=-1)

oe,X_train,X_test = encode(oe,X_train,X_test,object_cols)

models = {"LR":LogisticRegression(),
          "KNN":KNeighborsClassifier(),
          "DTC": DecisionTreeClassifier(),
          "RFC": RandomForestClassifier()}

pipes = []
for key in models.keys():
   pipe = pipeline(StandardScaler(), models[key],X_train,y_train)
   pipes.append(pipe)
   acc, recall, precision , conf = metrics(pipe,X_test,y_test)

   #print(f"{key}:\naccuracy : {acc}\nrecall : {recall} \nprecision : {precision} \n conf : {conf}")
""" Performance :
      
      LR:
        accuracy  : 0.874
        recall    : 0.383
        precision : 0.692
        conf      : [[239   8]
                     [ 29  18]]
      KNN:
        accuracy  : 0.854
        recall    : 0.170
        precision : 0.667
        conf      : [[243   4]
                     [ 39   8]]
      DTC:
        accuracy  : 0.779
        recall    : 0.319
        precision : 0.312
        conf      : [[214  33]
                     [ 32  15]]
      RFC:
        accuracy  : 0.840
        recall    : 0.106
        precision : 0.5
        conf      : [[242   5]
                     [ 42   5]]

accurate : LR(87.4)
precise  : LR(69.2)
recall   : LR(38.3)

"""

scores = validation(pipes[0],X_test,y_test,"recall",3)

df = pd.DataFrame(scores)



"""
validation:
    1     : 0.400
    2     : 0.375
    3     : 0.250
    mean  : 0.341

- Which is good that cross validation is almost equal to recall.
- So model is not overfitted.

"""

joblib.dump(pipes[0],"pipeline_LE.pkl")
joblib.dump(oe,"ordinal_encoder.pkl")

print("Model dumped successfuly")
