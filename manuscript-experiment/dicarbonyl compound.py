#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[2]:


#read
data_1 = pd.read_csv('../data/a1.csv', encoding = 'gbk')
data_2 = pd.read_csv('../data/c1.csv', encoding = 'gbk')
data_3 = pd.read_csv('../data/d1.csv', encoding = 'gbk')


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


data = data_8 - data_4 - data_6


# In[7]:


x_train,x_test, y_train, y_test = train_test_split(data,data_3.loc[:,'mol'],test_size=0.2,random_state=125)
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[11]:


print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[12]:


#RF
rf = RandomForestClassifier(n_estimators = 1000,
                       max_depth=None,min_samples_split=2,
                       random_state=0,class_weight='balanced').fit(x_train, y_train)
print(classification_report(y_test,  rf.predict(x_test)))
print('Accuracy on test set: {:.3f}'.format(rf.score(x_test, y_test)))
print("Accuracy on test: {}".format(rf.score(x_test, y_test)))


# In[13]:


#KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
print('Accuracy on test set: {:.3f}'.format(knn.score(x_test, y_test)))
print("Accuracy on test: {}".format(knn.score(x_test, y_test)))


# In[14]:


#SVM
svc = SVC(kernel='rbf', class_weight='balanced')
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
print('Accuracy on test set: {:.3f}'.format(svc.score(x_test, y_test)))
print("Accuracy on test: {}".format(svc.score(x_test, y_test)))


# In[ ]:




