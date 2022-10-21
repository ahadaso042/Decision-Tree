#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd

trainData = pd.read_csv('trainData.csv')
trainData


# In[8]:


trainData.dtypes


# In[9]:


testData = pd.read_csv('testData.csv')
testData



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(trainData['buying'])
trainData['buying'] = le.transform(trainData['buying'])
testData['buying'] = le.transform(testData['buying'])


# In[13]:


le2 = LabelEncoder()
le2.fit(trainData['target_class'])
trainData['target_class'] = le2.transform(trainData['target_class'])
testData['target_class'] = le2.transform(testData['target_class'])
trainData


# In[15]:


# Now, import a decision tree and fit it on the dataset
from sklearn import tree

tree1 = tree.DecisionTreeClassifier()
features = trainData['buying'].values.reshape(-1,1) # We need to reshape to have a list containing just one column.
tree1 = tree1.fit(features, trainData['target_class'])


# In[20]:


tree.plot_tree(tree1)


# In[21]:


predicted_1 = tree1.predict(testData['buying'].values.reshape(-1,1))


# In[22]:


# compare the predicted labels with actual ones
from sklearn.metrics import classification_report

print(classification_report(testData['target_class'].values.reshape((-1,1)), predicted_1,))


# ------------ Tree 2 ------------

# In[9]:


le3 = LabelEncoder()
le3.fit(trainData['persons'])
trainData['persons'] = le3.transform(trainData['persons'])
testData['persons'] = le3.transform(testData['persons'])


# In[10]:


le4 = LabelEncoder()
le4.fit(trainData['lug_boot'])
trainData['lug_boot'] = le4.transform(trainData['lug_boot'])
testData['lug_boot'] = le4.transform(testData['lug_boot'])


# In[11]:


le5 = LabelEncoder()
le5.fit(trainData['safety'])
trainData['safety'] = le5.transform(trainData['safety'])
testData['safety'] = le5.transform(testData['safety'])


# In[12]:


tree2 = tree.DecisionTreeClassifier()
tree2 = tree2.fit(trainData[['persons','lug_boot','safety']], trainData['target_class'])
tree.plot_tree(tree2)


# In[13]:


predicted_2 = tree2.predict(testData[['persons','lug_boot','safety']])


# In[14]:


print(classification_report(testData['target_class'].values.reshape((-1,1)), predicted_2,))


# ------------------- Tree 3 -------------------

# In[15]:


le6 = LabelEncoder()
le6.fit(trainData['maint'])
trainData['maint'] = le6.transform(trainData['maint'])
testData['maint'] = le6.transform(testData['maint'])


# In[16]:


le7 = LabelEncoder()
le7.fit(trainData['doors'])
trainData['doors'] = le7.transform(trainData['doors'])
testData['doors'] = le7.transform(testData['doors'])


# In[17]:


tree3 = tree.DecisionTreeClassifier()
tree3 = tree3.fit(trainData, trainData['target_class'])
tree.plot_tree(tree3)


# In[18]:


predicted_3 = tree3.predict(testData)


# In[19]:


print(classification_report(testData['target_class'].values.reshape((-1,1)), predicted_3,))


# In[ ]:




