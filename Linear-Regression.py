#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Data

# In[2]:


df=pd.read_csv("AMES_Final_DF.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


df.info()


# # TASK: The label we are trying to predict is the SalePrice column. Separate out the data into X features and y labels

# In[11]:


X=df.drop('SalePrice',axis=1)


# In[12]:


y = df['SalePrice']


# # TASK: Use scikit-learn to split up X and y into a training set and test set. Since we will later be using a Grid Search strategy, set your test proportion to 10%. To get the same data split as the solutions notebook, you can specify random_state = 101

# In[13]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# # TASK: The dataset features has a variety of scales and units. For optimal regression performance, scale the X features. Take carefuly note of what to use for .fit() vs what to use for .transform()

# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler=StandardScaler()


# In[18]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# # TASK: We will use an Elastic Net model. Create an instance of default ElasticNet model with scikit-learn

# In[19]:


from sklearn.linear_model import ElasticNet


# In[20]:


base_elastic_model = ElasticNet()


# # TASK: The Elastic Net model has two main parameters, alpha and the L1 ratio. Create a dictionary parameter grid of values for the ElasticNet. Feel free to play around with these values, keep in mind, you may not match up exactly with the solution choices

# In[21]:


param_grid = {'alpha':[0.1,1,5,10,50,100],
              'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}


# In[22]:


from sklearn.model_selection import GridSearchCV


# In[23]:


# verbose number a personal preference
grid_model = GridSearchCV(estimator=base_elastic_model,
                          param_grid=param_grid,
                          scoring='neg_mean_squared_error',
                          cv=5,
                          verbose=1)


# In[24]:


grid_model.fit(scaled_X_train,y_train)


# # TASK: Display the best combination of parameters for your model

# In[25]:


grid_model.best_params_


# # TASK: Evaluate your model's performance on the unseen 10% scaled test set. In the solutions notebook we achieved an MAE of $14149 and a RMSE of $20232

# In[26]:


y_pred = grid_model.predict(scaled_X_test)


# In[27]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[28]:


mean_absolute_error(y_test,y_pred)


# In[29]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[30]:


np.mean(df['SalePrice'])


# In[ ]:




