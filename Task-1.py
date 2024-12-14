#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report


# In[2]:


df = pd.read_csv('Dataset.csv.csv')
df


# # Data Preprocessing and Data Splitting

# In[3]:


df.isna().sum()


# In[4]:


df=df.dropna()


# In[5]:


df.isna().sum()


# In[6]:


#drop features that inhibit model building
df = df.drop('Restaurant ID', axis=1)
df = df.drop('Restaurant Name', axis=1)
df = df.drop('Country Code', axis=1)
df = df.drop('City', axis=1)
df = df.drop('Address', axis=1)
df = df.drop('Locality', axis=1)
df = df.drop('Locality Verbose', axis=1)
df = df.drop('Longitude', axis=1)
df = df.drop('Latitude', axis=1)
df = df.drop('Cuisines', axis=1)
df = df.drop('Currency', axis=1)


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df


# In[10]:


#encode yes-No labels of categorical festures int binary (1 for yes and 0 for no)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Has Table booking'] = le.fit_transform(df['Has Table booking'])
df['Has Online delivery'] = le.fit_transform(df['Has Online delivery'])
df['Is delivering now'] = le.fit_transform(df['Is delivering now'])
df['Switch to order menu'] = le.fit_transform(df['Switch to order menu'])
df['Rating color'] = le.fit_transform(df['Rating color'])
df['Rating text'] = le.fit_transform(df['Rating text'])


# In[11]:


df


# In[12]:


df['Aggregate rating'].value_counts().plot(kind='pie', autopct = '%.3f')


# In[13]:


sns.distplot(df['Aggregate rating'])


# In[14]:


sns.scatterplot(x=df["Aggregate rating"], y=df["Votes"], hue=df["Price range"]) 


# In[15]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.title("Correleation between the attributes")
plt.show()


# In[16]:


x = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']


# In[17]:


#data splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=250)
x_train.head()
y_train.head()


# In[18]:


print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# # Running Linear regression Model

# In[19]:


#training by linear regression model
linreg = LinearRegression()
linreg.fit(x_train,y_train)
linreg_pred = linreg.predict(x_test)


# In[20]:


#evaluating performance metrics of linear regression
linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_mse = mean_squared_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)
print(f"MAE of the linear regression model is: {linreg_mae:.2f}")
print(f"MSE of the linear regression model is: {linreg_mse:.2f}")
print(f"R2 score of the linear regression model is: {linreg_r2:.2f}")


# # Running Decision Tree

# In[21]:


# training by decision tree regressor algorithm
dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)


# In[22]:


#evaluating performance metrices of liner regression
dtree_mae = mean_absolute_error(y_test, dtree_pred)
dtree_mse = mean_squared_error(y_test, dtree_pred)
dtree_r2 = r2_score(y_test, dtree_pred)
print(f"MAE of the decision tree model is: {dtree_mae:.2f}")
print(f"MSE of the decision tree model is: {dtree_mse:.2f}")
print(f"R2 score of the decision tree model is: {dtree_r2:.2f}")


# # Model achieves 98% accuracy

# MSE of 0.05 indicates that model's predictions are very accurate & low errors.
# R2 value of 0.98 suggests that model is highly effective at explaining & predicting the target variable.
# Decision Tree Regressor model is performing exceptionally well on the test data

# # Analysing the factors affecting restaurent rating

# Distribution of the target variable ("Aggregate rating") is well balanced.
# Expensive restaurants (higher price range) tend to have higher ratings.
