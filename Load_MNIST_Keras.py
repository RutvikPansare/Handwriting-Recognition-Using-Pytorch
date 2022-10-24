#!/usr/bin/env python
# coding: utf-8

# In[23]:


from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('pinfo', 'mnist.load_data')


# In[14]:


(X_train,y_train),(X_test,y_test) = mnist.load_data()


# In[15]:


X_train.shape 


# In[16]:


X_test.shape


# In[17]:


y_train.shape


# In[18]:


y_test.shape


# In[24]:


plt.imshow(X_train[0])


# In[ ]:




