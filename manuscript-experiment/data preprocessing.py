#!/usr/bin/env python
# coding: utf-8

# In[1]:
# out[2]

#preprocessing
import numpy as np
import pandas as pd


# In[2]:


#read
data_1 = pd.read_csv('C:/Users/86157/yang/a1.csv', encoding = 'gbk')
data_2 = pd.read_csv('C:/Users/86157/yang/c1.csv', encoding = 'gbk')
data_3 = pd.read_csv('C:/Users/86157/yang/AGE1.csv', encoding = 'gbk')
fl = pd.read_csv('C:/Users/86157/yang/mz1.csv', encoding = 'gbk')


# In[3]:


# Drop null rows
data_4 = data_1.drop(['mol'], axis = 'columns')
data_5 = data_1.apply(pd.to_numeric, errors = 'ignore')
data_5.isnull().sum()
data_del_NAN_1 = data_5.dropna(axis=0)
data_del_NAN_1 = pd.DataFrame(data_del_NAN_1)


# In[4]:


data_6 = data_2.drop(['mol'], axis = 'columns')
data_7 = data_2.apply(pd.to_numeric, errors = 'ignore')
data_7.isnull().sum()
data_del_NAN_2 = data_7.dropna(axis=0)
data_del_NAN_2 = pd.DataFrame(data_del_NAN_2)


# In[5]:


data_8 = data_3.drop(['mol'], axis = 'columns')
data_9 = data_3.apply(pd.to_numeric, errors = 'ignore')
data_9.isnull().sum()
data_del_NAN_3 = data_9.dropna(axis=0)
data_del_NAN_3 = pd.DataFrame(data_del_NAN_3)


# In[6]:


data_10 = data_4 + data_6
data_11 = data_8 - data_10


# In[7]:


file = pd.concat([data_11,fl],axis=1,ignore_index=False)
file.to_csv('data.csv')


# In[ ]:




