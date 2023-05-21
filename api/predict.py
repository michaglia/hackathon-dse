#!/usr/bin/env python
# coding: utf-8

# # Local predictions with FastAPI

# In[1]:


import requests
import pandas as pd


# ### Load data

# In[2]:


winedf = pd.read_csv("C:/Users/secon/Desktop/Hackathon/winequality-red.csv")


# ### Convert to json

# In[3]:


X = winedf.iloc[:, 0:-1]
json = [X.iloc[i].to_dict() for i in range(X.shape[0])]


# ### Predict

# In[4]:

url = 'http://127.0.0.1:8000/predict/'
res = requests.post(url, json=json)
y_pred = res.json()
print(y_pred[0:10])

