import pandas as pd
import numpy as np 
import seaborn as sns

df = pd.read_csv("observed_data.csv")

columns = ['Age', 
           'Attrition', 
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
           'Over18', 
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
           ]


for col in df.columns:
    if df[col].dtypes == "object":
        print(df[col].value_counts())

""" 
Conclusion:
       - max values are 9 
       - most of columns have less values
       - Over18 is useless columns(only 1 value yes)
"""

df = df.drop("Over18",axis=1)

df.to_csv("cleaned_data.csv",index=False)



