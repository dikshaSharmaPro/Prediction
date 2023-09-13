#!/usr/bin/env python
# coding: utf-8

# In[2]:


#rainfall prediction model


# In[3]:


import pandas as pd


# In[4]:


import pandas as np


# In[5]:


import math


# In[6]:


import sklearn


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing


# In[12]:


data=pd.read_csv("rainfall.csv")


# In[13]:


print("Data heads:")
print(data.head())
print("Null values in the dataset before preprocessing:")
print(data.isnull().sum())
print("Filling null values with mean of that particular column")
data=data.fillna(np.mean(data))
print("Mean of data:")
print(np.mean(data))
print("Null values in the dataset after preprocessing:")
print(data.isnull().sum())
print("\n\nShape: ",data.shape)


# In[14]:


print("Info:")
print(data.info())


# In[15]:


print("Group by:")
data.groupby('SUBDIVISION').size()


# In[16]:


print("Co-Variance =",data.cov())
print("Co-Relation =",data.corr())


# In[17]:


corr_cols=data.corr()['ANNUAL'].sort_values()[::-1]
print("Index of correlation columns:",corr_cols.index)


# In[18]:


print("Scatter plot of annual and january attributes")
plt.scatter(data.ANNUAL,data.JAN)


# In[19]:


print("Box Plot of annual rainfall data in years 1901-2015")
data['ANNUAL'].plot(kind='box',sharex=False,sharey=False)


# In[20]:


print("Histograms showing the data from attributes (JANUARY to DECEMBER) of the years 1901-2015:")
data['JAN'].hist(bins=20)
data['FEB'].hist(bins=20)
data['MAR'].hist(bins=20)
data['APR'].hist(bins=20)
data['MAY'].hist(bins=20)
data['JUN'].hist(bins=20)
data['JUL'].hist(bins=20)
data['AUG'].hist(bins=20)
data['SEP'].hist(bins=20)
data['OCT'].hist(bins=20)
data['NOV'].hist(bins=20)
data['DEC'].hist(bins=20)


# In[21]:


print("Histogram showing the annual rainfall of the all states:")
data['ANNUAL'].hist(bins=20)


# In[22]:


print("Violin plot of the ANNUAL attribute :-")
sb.violinplot(data=data['ANNUAL'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




