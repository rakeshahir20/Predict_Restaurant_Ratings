#!/usr/bin/env python
# coding: utf-8

# # Restaurant Recommendation

# In[1]:


import numpy as npy
import pandas as pdf
import seaborn as sborn
import matplotlib.pyplot as plott


# In[2]:


from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform


# In[3]:


pdf.reset_option('display.max_rows')


# In[4]:


dframe=pdf.read_csv('Dataset.csv.csv')


# In[5]:


dframe.head()


# In[6]:


dframe.columns


# In[7]:


dfRS = dframe[['Restaurant ID','Restaurant Name','Cuisines','Aggregate rating','Votes']]
dfRS


# In[8]:


# Columns Description
def dataDesc():
    listItem = []
    for col in dfRS.columns :
        listItem.append([col,dfRS[col].dtype,dfRS[col].isna().sum(),round(dfRS[col].isna().sum()/len(dfRS)*100,2),dfRS[col].nunique(),list(dfRS[col].drop_duplicates().sample(2).values)])
    descData = pdf.DataFrame(data = listItem,columns = ['Column','Data Type', 'Missing Value','Pct Missing Value', 'Num Unique', 'Unique Sample'])
    return descData

dataDesc()


# In[9]:


dfRS = dfRS.dropna()


# In[10]:


dfRS


# In[11]:


dfRS = dfRS.rename(columns={'Restaurant ID': 'restaurant_id'})
dfRS = dfRS.rename(columns={'Restaurant Name': 'restaurant_name'})
dfRS = dfRS.rename(columns={'Cuisines': 'cuisines'})
dfRS = dfRS.rename(columns={'Aggregate rating': 'aggregate_rating'})
dfRS = dfRS.rename(columns={'Votes': 'votes'})
dfRS


# In[12]:


dfRS.duplicated().sum()


# In[13]:


dfRS['restaurant_name'].duplicated().sum()


# In[14]:


dfRS['restaurant_name'].value_counts()


# In[15]:


dfRS = dfRS.sort_values(by=['restaurant_name','aggregate_rating'],ascending=False)
dfRS[dfRS['restaurant_name']=="Domino's Pizza"].head()


# In[16]:


dfRS = dfRS.drop_duplicates('restaurant_name',keep='first')
dfRS


# In[17]:


dfRS['restaurant_name'].value_counts()


# In[18]:


# Cross Tabulate Restaurant Name and Cuisines

xTabRestoCuisines = pdf.crosstab(dfRS['restaurant_name'],dfRS['cuisines'])
xTabRestoCuisines


# In[19]:


# Checking on restaurant name value
xTabRestoCuisines.loc['feel ALIVE'].values


# In[20]:


dfRS['restaurant_name'].sample(20, random_state=101)


# In[21]:


# Measure Similarity

print(jaccard_score(xTabRestoCuisines.loc["Olive Bistro"].values,xTabRestoCuisines.loc["Rose Cafe"].values))


# In[ ]:


# Create Similarity Value DF

jaccardDist = pdist(xTabRestoCuisines.values, metric='jaccard')
jaccardMatrix = squareform(jaccardDist)
jaccardSim = 1 - jaccardMatrix
dfJaccard = pdf.DataFrame(jaccardSim,index=xTabRestoCuisines.index,columns=xTabRestoCuisines.index)

dfJaccard


# In[ ]:


# Resto Names Sample

dfRS['restaurant_name'].sample(20)


# In[ ]:


# Make Recomendation

# Input Initial Restaurant Name
resto = 'Ooma'

sim = dfJaccard.loc[resto].sort_values(ascending=False)

sim = pdf.DataFrame({'restaurant_name': sim.index, 'simScore': sim.values})
sim = sim[(sim['restaurant_name']!= resto) & (sim['simScore']>=0.7)].head(5)

# Merge The Rating

RestoRec = pdf.merge(sim,dfRS[['restaurant_name','aggregate_rating']],how='inner',on='restaurant_name')
FinalRestoRec = RestoRec.sort_values('aggregate_rating',ascending=False).drop_duplicates('restaurant_name',keep='first')
FinalRestoRec


# In[ ]:




