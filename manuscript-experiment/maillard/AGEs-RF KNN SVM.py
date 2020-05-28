#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pickle
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[2]:


data_1 = pd.read_csv('C:/Users/86157/yang/a1.csv', encoding = 'gbk')
data_2 = pd.read_csv('C:/Users/86157/yang/c1.csv', encoding = 'gbk')
data_3 = pd.read_csv('C:/Users/86157/yang/AGE1.csv', encoding = 'gbk')


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
data = data_8 - data_10
data.keys()
labels = list(data.columns.values)
print(labels)


# In[7]:


#利用StandScaler缩放数据
scaler = StandardScaler()
scaler.fit(data)
X_scaled = scaler.transform(data)


# In[8]:


#构建PCA模型
pca = PCA(n_components = 2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print("Original shape:{}".format(str(X_scaled.shape)))
print("Reduced shape:{}".format(str(X_pca.shape)))


# In[9]:


plt.scatter(X_pca[:, 0], X_pca[:, 1], marker = '>', c = 'r')
# 设置坐标标签
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
# 设置标题
plt.title("PCA Scatter Plot")

# 显示图形
plt.show()


# In[10]:


print(pca.explained_variance_ratio_)
print(pca.singular_values_)


# In[11]:


print("PCA compent shape:{}".format(pca.components_.shape))


# In[12]:


print("PCA compents:\n{}".format(pca.components_))


# In[43]:


x_train,x_test, y_train, y_test = train_test_split(data,data_3.loc[:,'mol'],test_size=0.2, random_state=60)
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[44]:


#RF
rf = RandomForestClassifier(n_estimators = 1000,
                       max_depth=None,min_samples_split=2,
                       random_state=0,class_weight='balanced').fit(x_train, y_train)
print(classification_report(y_test,  rf.predict(x_test)))
print('Accuracy on test set: {:.3f}'.format(rf.score(x_test, y_test)))
print("Accuracy on test: {}".format(rf.score(x_test, y_test)))


# In[45]:


#KNN
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
print('Accuracy on test set: {:.3f}'.format(knn.score(x_test, y_test)))
print("Accuracy on test: {}".format(knn.score(x_test, y_test)))


# In[46]:


#SVM
svc = SVC(kernel='rbf', class_weight='balanced')
svc.fit(x_train, y_train)
y_predict = svc.predict(x_test)
print('Accuracy on test set: {:.3f}'.format(svc.score(x_test, y_test)))
print("Accuracy on test: {}".format(svc.score(x_test, y_test)))


# In[ ]:




