import pandas as pd 
import numpy as np
import seaborn as sns

df = pd.read_csv("employee_sat.csv")

cols = ['Age', 'Attrition', 'BusinessTravel',
        'DailyRate', 'Department', 'DistanceFromHome', 
        'Education', 'EducationField', 'EmployeeCount',                     'EmployeeNumber', 'EnvironmentSatisfaction', 
        'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel',
        'JobRole', 'JobSatisfaction', 'MaritalStatus',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'Over18', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction',
        'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]

train = df[df["EmployeeNumber"]<=2000]
to_test = df[df["EmployeeNumber"]>2000]

to_test.to_csv("test.csv",index=False)

print(to_test.shape)

shape = train.shape
size = train.size
""" data has 1470 rows which not big compare to other data
    but total columns are identical(35)
"""
                
train.info()

""" Conclusion:
       - No null values
       - 9 object type values 
"""

stat = train.describe()

""" Conclusion:
       - we definitely need scaling
"""

dups = train.duplicated().sum()
""" Conclusion:
       - No duplicate rows
"""

targets = train.Attrition.value_counts()

"""Attrition. No     1233. Yes     237
so this is completely imbalenced target if model predict all No accuracy will be still 0.838 looks good but it would be worst model

conclusion: 
* we need recall score, precision score , confusion matrix to judge the model 
* we need to stratify y during splitting """

train["Attrition"] = (train["Attrition"]=="Yes").astype(int)

""" 
-Employee count is a useless column because it has only one value
-Employee Number is like id
"""
train = train.drop(columns=["EmployeeCount","EmployeeNumber"])

train.to_csv("observed_data.csv",index=False)


