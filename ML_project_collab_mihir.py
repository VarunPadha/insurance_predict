#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt


# In[2]:


Insurance=pd.read_csv("insurance.csv")
Insurance.describe()


# In[3]:


Insurance.value_counts()


# In[4]:


sns.boxplot(Insurance)


# In[5]:


sns.heatmap(Insurance.describe())


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
Insurance.hist(bins=50,figsize=(20,15))


# In[7]:


Insurance.info()


# In[8]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(Insurance,test_size=0.2,random_state= 42)
print(f"train set {train_set}")
print(f"test set {test_set}")


# In[9]:


Insurance_corr=Insurance.corr()
Insurance_corr['charges'].sort_values(ascending=False)


# In[10]:


from pandas.plotting import scatter_matrix
scatter_matrix(Insurance,figsize=(12,15))


# In[11]:


Insurance=train_set.drop("charges",axis=1)
Insurance_label=train_set["charges"].copy()


# In[12]:


Datamap_sex={
    "female":0,
    "male":1,
}
Insurance["sex"]=Insurance["sex"].map(Datamap_sex)
Datamap_smoker={
    "yes":0,
    "no":1,
}
Insurance["smoker"]=Insurance["smoker"].map(Datamap_smoker)
Datamap_region={
    "northwest":0,
    "southwest":1,
}
Insurance["region"]=Insurance["region"].map(Datamap_region)


# In[13]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
insurance_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('Standerdization',StandardScaler()),
    ])


# In[14]:


Insurance_stand=insurance_pipeline.fit_transform(Insurance)


# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(Insurance_stand,Insurance_label)


# In[16]:


from sklearn.metrics import mean_squared_error
Insurance_predict=model.predict(Insurance_stand)
error=mean_squared_error(Insurance_predict,Insurance_label)
np.sqrt(error)


# In[17]:


from sklearn.model_selection import cross_val_score

scores=cross_val_score(model,Insurance_stand,Insurance_label,scoring="neg_mean_squared_error")


# In[18]:


np.sqrt(-scores)


# In[ ]:




