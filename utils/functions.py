from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

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
                             


def split(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.2,
                                                 random_state=42,
                                                 stratify=y)
    return X_train,X_test,y_train,y_test


def pipeline(scaler,model,X_train,y_train):
    pipe = Pipeline([
       ("scaler", scaler),
       ("model",model)
       ])
    pipe.fit(X_train,y_train)
    
    return pipe
    

def metrics(pipe,X_test,y_test):
    
    y_pred = pipe.predict(X_test)
    
    ac = accuracy_score(y_test,y_pred)
    rs = recall_score(y_test,y_pred)
    ps = precision_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    return ac,rs,ps,cm


def encode(oe,X_train,X_test,cols):
     X_train[cols] = oe.fit_transform(X_train[cols])
     X_test[cols] = oe.transform(X_test[cols])
     
     return oe,X_train,X_test


def validation(pipe,X_test,y_test,scoring,cv):
     score = cross_val_score(pipe,X_test,y_test, scoring=scoring ,cv=cv)
     return score
     
     
def pipe_with_column_transform(scaler,encoder,model,num,cat,X_train,y_train):
     preprocessor = ColumnTransformer([
                     ("cat",encoder,cat),
                     ("num",scaler,num)
                   ])
     pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model",model)
           ])
     pipe.fit(X_train,y_train)
     return pipe
     

