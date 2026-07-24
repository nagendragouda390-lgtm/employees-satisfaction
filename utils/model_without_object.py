import pandas as pd
import numpy as np 
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score,
                             recall_score, 
                             precision_score, 
                             confusion_matrix)

from functions import * 

df = pd.read_csv("cleaned_data.csv")

for col in df.columns:
    if df[col].dtypes == "object":
        df = df.drop(col,axis=1)

X = df.drop("Attrition",axis=1)
print(X.columns)
y = df["Attrition"]

"""
without stratify:
    train : 
       0    978
       1    198
       ratio : 4.93
    test  :
       0    255
       1     39
       ratio : 6.53
with stratify:
    
    train :
       0    986
       1    190
       ratio : 5.18
    test  : 
       0    247
       1     47
       ratio : 5.23

"""
        
X_train,X_test,y_train,y_test = split(X,y)

pipe = pipeline(StandardScaler(),
                LogisticRegression(),
                X_train,
                y_train)

pipe2 = pipeline(StandardScaler(),
                RandomForestClassifier(random_state=43),
                X_train,
                y_train)

pipe3 = pipeline(StandardScaler(),
                KNeighborsClassifier(),
                X_train,
                y_train)
                
pipe4 = pipeline(StandardScaler(),
                DecisionTreeClassifier(),
                X_train,
                y_train)
               
accuracy, recall, precision,matrix = metrics(pipe,X_test,y_test)

#print(f"\nLR  :\naccuracy : {accuracy}\nrecall : {recall}\nprecision : {precision}\nmatrix : {matrix}")


accuracy2, recall2, precision2,matrix2 = metrics(pipe2,X_test,y_test)

#print(f"\nRFC : \naccuracy : {accuracy2}\nrecall : {recall2}\nprecision : {precision2}\nmatrix : {matrix2}")

accuracy3, recall3, precision3,matrix3 = metrics(pipe3,X_test,y_test)

#print(f"\nKNN : \naccuracy : {accuracy3}\nrecall : {recall3}\nprecision : {precision3}\nmatrix : {matrix3}")

accuracy4, recall4, precision4,matrix4 = metrics(pipe4,X_test,y_test)

#print(f"\nDTC : \naccuracy : {accuracy4}\nrecall : {recall4}\nprecision : {precision4}\nmatrix : {matrix4}")

"""
LR  :
    accuracy  : 0.857
    recall    : 0.234
    precision : 0.647
    matrix    : [[241   6]
                 [ 36  11]]

RFC :
    accuracy  : 0.823
    recall    : 0.085
    precision : 0.307
    matrix    : [[238   9]
                 [ 43   4]]

KNN :
    accuracy  : 0.829
    recall    : 0.085
    precision : 0.363
    matrix    : [[240   7]
                 [ 43   4]]

DTC :
    accuracy  : 0.721
    recall    : 0.234
    precision : 0.193
    matrix    : [[201  46]
                 [ 36  11]]

most_accurate = LR(0.857)
most_recall   = LR and DTC(0.234)
most_precise  = LR(0.647)
conf_matrix   = 

so better model is LogisticRegression.

"""
import joblib

joblib.dump(pipe,"LR_without_object.pkl")





