import pandas as pd
import numpy as np
import joblib


df = pd.read_csv("test.csv")

test = df.drop("Attrition",axis=1)
model_without_object = joblib.load("LR_without_object.pkl")

model_label_encoder = joblib.load("pipeline_LE.pkl")

model_one_hot_encoder = joblib.load("pipe_ohe.pkl")

le = joblib.load("label_encoder.pkl")

df_w_o = test[[
             'Age',
             'DailyRate',
             'DistanceFromHome',
             'Education',
             'EnvironmentSatisfaction', 
             'HourlyRate',
             'JobInvolvement', 
             'JobLevel',
             'JobSatisfaction',
             'MonthlyIncome', 
             'MonthlyRate', 
             'NumCompaniesWorked',
             'PercentSalaryHike', 
             'PerformanceRating',
             'RelationshipSatisfaction',
             'StandardHours', 
             'StockOptionLevel', 
             'TotalWorkingYears',
             'TrainingTimesLastYear', 
             'WorkLifeBalance', 
             'YearsAtCompany',
             'YearsInCurrentRole', 
             'YearsSinceLastPromotion',
             'YearsWithCurrManager'
             ]]

prediction_1 = model_without_object.predict(df_w_o)

object_cols = []

for col in test.columns:
    if test[col].dtypes == "object":
        object_cols.append(col)

df_le = test[[
             'Age',
             'BusinessTravel', 
             'DailyRate', 
             'Department', 
             'DistanceFromHome',
             'Education', 
             'EducationField', 
             'EnvironmentSatisfaction', 
             'Gender',
             'HourlyRate', 
             'JobInvolvement', 
             'JobLevel', 
             'JobRole',
             'JobSatisfaction', 
             'MaritalStatus', 
             'MonthlyIncome', 
             'MonthlyRate',
             'NumCompaniesWorked', 
             'OverTime', 
             'PercentSalaryHike',
             'PerformanceRating', 
             'RelationshipSatisfaction', 
             'StandardHours',
             'StockOptionLevel',
             'TotalWorkingYears',
             'TrainingTimesLastYear',
             'WorkLifeBalance',
             'YearsAtCompany',
             'YearsInCurrentRole',
             'YearsSinceLastPromotion', 
             'YearsWithCurrManager'
           ]]
for col in object_cols:
      df_le[col] = le.transform(df_le[col])

predictions_2 = model_label_encoder(df_le)

print(predictions_2)

