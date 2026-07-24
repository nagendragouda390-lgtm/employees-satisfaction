import pandas as pd
import numpy as np
import joblib


df = pd.read_csv("test.csv")

test = df.drop("Attrition",axis=1)
model_without_object = joblib.load("LR_without_object.pkl")

model_label_encoder = joblib.load("pipeline_LE.pkl")

model_one_hot_encoder = joblib.load("pipe_ohe.pkl")

oe = joblib.load("ordinal_encoder.pkl")

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

print(prediction_1)
object_cols = []

for col in test.columns:
    if col != "Over18":
      if test[col].dtypes == "object":
         object_cols.append(col)
 
df_oe = test[[
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

df_oe[object_cols] = oe.transform(df_oe[object_cols])

prediction_2 = model_label_encoder.predict(df_oe)

print(prediction_2)

df_ohe = df[[
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

prediction_3 = model_one_hot_encoder.predict(df_ohe)

print(prediction_3)

result = pd.DataFrame({
         "actual": df["Attrition"],
         "no_encoder": prediction_1,
         "OrdinalEncoder" : prediction_2,
         "OneHotEncoder" : prediction_3,
})

result.to_csv("result.csv",index=False)