#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as nump
import pandas as pdf


# In[2]:


dframe=pdf.read_csv('Dataset.csv.csv')


# In[3]:


dframe.drop('Restaurant ID', axis=1, inplace=True)
dframe.drop('Country Code', axis=1, inplace=True)
dframe.drop('City', axis=1, inplace=True)
dframe.drop('Address', axis=1, inplace=True)
dframe.drop('Locality', axis=1, inplace=True)
dframe.drop('Locality Verbose', axis=1, inplace=True)
dframe.drop('Longitude', axis=1, inplace=True)
dframe.drop('Latitude', axis=1, inplace=True)
dframe.drop('Currency', axis=1, inplace=True)
dframe.drop('Has Table booking', axis=1, inplace=True)
dframe.drop('Has Online delivery', axis=1, inplace=True)
dframe.drop('Is delivering now', axis=1, inplace=True)
dframe.drop('Switch to order menu', axis=1, inplace=True)
dframe.drop('Price range', axis=1, inplace=True)
dframe.drop('Aggregate rating', axis=1, inplace=True)
dframe.drop('Rating color', axis=1, inplace=True)
dframe.drop('Rating text', axis=1, inplace=True)
dframe.drop('Votes', axis=1, inplace=True)


# In[4]:


dframe


# In[5]:


dframe.isnull().sum()


# In[6]:


dframe.dropna(inplace=True)


# In[7]:


dframe.shape


# In[8]:


missing_values = dframe.isna().sum()
missing_values_column = dframe['Restaurant Name'].isna().sum()
missing_values_column = dframe['Cuisines'].isna().sum()
missing_values_column = dframe['Average Cost for two'].isna().sum()


# In[9]:


df_cleaned = dframe.dropna()
df_cleaned = dframe.dropna(subset=['Restaurant Name'])
df_cleaned = dframe.dropna(subset=['Cuisines'])
df_cleaned = dframe.dropna(subset=['Average Cost for two'])


# In[10]:


dframe.describe()


# In[11]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dframe['Restaurant Name'] = label_encoder.fit_transform(dframe['Restaurant Name'])
dframe['Cuisines'] = label_encoder.fit_transform(dframe['Cuisines'])


# In[12]:


dframe


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[14]:


X = dframe[['Restaurant Name', 'Average Cost for two']]
y = dframe['Cuisines']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# In[17]:


rf_classifier.fit(X_train, y_train)


# In[18]:


y_pred = rf_classifier.predict(X_test)


# In[19]:


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[20]:


print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[22]:


X = dframe[['Restaurant Name', 'Average Cost for two']]
y = dframe['Cuisines']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[25]:


model.fit(X_train, y_train)


# In[26]:


y_pred = model.predict(X_test)


# In[27]:


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[28]:


print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)

