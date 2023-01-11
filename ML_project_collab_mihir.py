#!/usr/bin/env python
# coding: utf-8




import numpy as np 
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt





Insurance=pd.read_csv("insurance.csv")
Insurance.describe()





Insurance.value_counts()





sns.boxplot(Insurance)





sns.heatmap(Insurance.describe())





get_ipython().run_line_magic('matplotlib', 'inline')
Insurance.hist(bins=50,figsize=(20,15))





Insurance.info()





from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(Insurance,test_size=0.2,random_state= 42)
print(f"train set {train_set}")
print(f"test set {test_set}")





Insurance_corr=Insurance.corr()
Insurance_corr['charges'].sort_values(ascending=False)




from pandas.plotting import scatter_matrix
scatter_matrix(Insurance,figsize=(12,15))





Insurance=train_set.drop("charges",axis=1)
Insurance_label=train_set["charges"].copy()





from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
Insurance_new=ohe.fit_transform(Insurance[['sex','region','smoker']]).toarray()
Insurance=np.hstack((Insurance[['index','bmi','age','children']].values,Insurance_new))




from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
insurance_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('Standerdization',StandardScaler()),
    ])





Insurance_stand=insurance_pipeline.fit_transform(Insurance)





from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(Insurance_stand,Insurance_label)





from sklearn.metrics import mean_squared_error
Insurance_predict=model.predict(Insurance_stand)
error=mean_squared_error(Insurance_predict,Insurance_label)
np.sqrt(error)





from sklearn.model_selection import cross_val_score

scores=cross_val_score(model,Insurance_stand,Insurance_label,scoring="neg_mean_squared_error")





np.sqrt(-scores)

from joblib
joblib.dump(model,"insurance_predection)





