#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler() 
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline


# In[19]:


data_wine = pd.read_csv('C:/Users/Michela/Downloads/archive (1)/winequality-red.csv')


# In[20]:


data_wine


# In[21]:


data_wine.isnull().sum()


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


data_wine['quality'].hist(color='red')
plt.title('Quality of wine')


# In[24]:


quality = data_wine['quality'].value_counts()
quality


# In[25]:


# prepare data for fitting: if wine has quality over 5 it is a good wine; if it has belove it is a bad wine
# changing the values to Boolean operators (True or False)


# In[26]:


good_wine = data_wine['quality']>5
good_wine


# In[27]:


import numpy as np
wine_rate = np.array(data_wine['quality']>5)
mapping = {True: 'Good wine', False: 'Bad wine'}
y = np.vectorize(mapping.get)(wine_rate)
y


# In[28]:


X_variables =data_wine.iloc[:,:-1]

X_variables


# ## Train and split 

# In[29]:


from sklearn.model_selection import train_test_split, GridSearchCV

X_variables_train,X_variables_test,y_train,y_test=train_test_split(X_variables,y,random_state=42,test_size=0.2)


# In[31]:


from sklearn.svm import SVC
ensamble = VotingClassifier(estimators=[
    ('mnb', MultinomialNB()),
    ('svc', SVC()),
    ('rf', RandomForestClassifier())
])


# In[32]:


pipe = Pipeline([
    ('encoder', None), 
    ('classifier', ensamble),
])


# In[33]:


cls = GridSearchCV(
    pipe, 
    {
        'encoder': [
            None
        ],
        'classifier__mnb__alpha': [0.1, 1, 2],
        'classifier__svc__C': [0.1, 1, 10],
        'classifier__svc__class_weight': ['balanced'],
        'classifier__rf__n_estimators': [10, 100],
        'classifier__rf__criterion': ['gini', 'entropy'],
    }, 
    cv=5, 
    scoring='f1_macro'
)


# In[34]:


scaler_X_variables_train=scaler.fit_transform(X_variables_train)

scaler_X_variables_train


# In[35]:


scaler_X_variables_test=scaler.transform(X_variables_test)

scaler_X_variables_test


# In[36]:


cls.fit(X_variables_train, y_train)
cls.best_params_


# In[37]:


print('Validation score', cls.best_score_)
print('Test score', cls.score(X_variables_test, y_test))


# In[39]:


pickle.dump(cls, open('model.pkl', 'wb'))


# ## Building SVM Classifier

# In[28]:


from sklearn.svm import SVC

svc_classifier = SVC(C=1.0, 
              kernel='rbf', 
              degree=3, 
              gamma='auto', 
              coef0=0.0, shrinking=True, 
              probability=False, 
              tol=0.001, cache_size=200, 
              class_weight=None, 
              verbose=False, max_iter=-1, 
              decision_function_shape='ovr', 
              break_ties=False,random_state=None)

svc_classifier.fit(scaler_X_variables_train,good_wine_train)


# In[29]:


svc_classifier_predictions=svc_classifier.predict(scaler_X_variables_test)


# In[30]:


c=confusion_matrix(good_wine_test,svc_classifier_predictions)
a=accuracy_score(good_wine_test,svc_classifier_predictions)
p=precision_score(good_wine_test,svc_classifier_predictions)
r=recall_score(good_wine_test,svc_classifier_predictions)


# In[32]:


print('Accuracy', a*100)
print('Precision', p*100)
print('Recall score', r*100)


# In[34]:


print('Confusion Matrix\n', c)

