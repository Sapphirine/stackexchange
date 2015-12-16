
# coding: utf-8

# In[143]:

import pandas as pd
import numpy as np


# In[144]:

df = pd.read_csv('/Users/XW/Desktop/datascience.stackexchange.com/result.csv',header = None)


# In[145]:

#df.shape


# In[146]:

df2 = pd.read_csv('/Users/XW/Desktop/datascience.stackexchange.com/matrix0.csv')


# In[147]:

words = list(df2.columns.values)


# In[150]:

topics = pd.DataFrame()    
for i in range(10):
    flag = df.sort_index(by=[i], ascending=[False])
    value = flag[i][0:10]
    Ind = flag.index.values[:10]
    Words = [words[i] for i in Ind]
    topics[2*i] = pd.Series(Words)
    topics[2*i+1] = pd.Series(np.array(value))


# In[151]:

topics


# In[ ]:



