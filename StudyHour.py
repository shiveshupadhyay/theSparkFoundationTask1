#!/usr/bin/env python
# coding: utf-8

# # Shivesh Upadhyay

# # Data Pre-processing and Visualisation
#    In this section we make aur data ready for applying regressor model by assigning dependent and independent variable

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


dataset


# In[4]:


print(X)


# In[5]:


print(y)


# In[6]:


plt.scatter(X,y)


# # Data Training and Testing
#    Here we split data in training and testing set, train the model and predict values for testing set.

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[8]:


print(X_train)


# In[9]:


print(X_test)


# In[10]:


print(y_train)


# In[11]:


print(y_test)


# In[12]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[13]:


y_pred = regressor.predict(X_test)


# # Visualising Testing and Predicted Data
#    Here we visualise our prediction in comaprison to  given data

# ### Visualising for training data

# In[14]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Percentage as per study hour')
plt.xlabel('Study hour')
plt.ylabel('Percentage')
plt.show()


# ### Visualising for testing data

# In[15]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Percentage as per study hour')
plt.xlabel('Study hour')
plt.ylabel('Percentage')
plt.show()


# ### Predicting for 9.25 hours

# In[16]:


regressor.predict([[9.25]])


# ### Calculating error 

# In[17]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

