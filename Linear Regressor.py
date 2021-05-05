#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


# In[2]:


dataset = pd.read_csv('D:\CE888_Data Science_Decision Making\Model_Deployment\Breast_cancer_data.csv')


# In[4]:


x = dataset[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
       'mean_smoothness']]
y = dataset['diagnosis']


# In[7]:


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
regressor = LinearRegression()


# In[8]:


#Fitting model with trainig data
regressor.fit(x, y)


# In[12]:


# Saving model to disk
pickle.dump(regressor, open('D:\CE888_Data Science_Decision Making\Model_Deployment\model.pkl','wb'))


# In[13]:


# Loading model to compare the results
model = pickle.load(open('D:\CE888_Data Science_Decision Making\Model_Deployment\model.pkl','rb'))

