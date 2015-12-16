
# coding: utf-8

# In[101]:

import pandas as pd


# In[102]:

df = pd.read_csv('/Users/XW/Desktop/datascience.stackexchange.com/result.csv',header = None)


# In[103]:

#df.shape


# In[104]:

df2 = pd.read_csv('/Users/XW/Desktop/datascience.stackexchange.com/matrix0.csv')


# In[105]:

words = list(df2.columns.values)


# In[106]:

topics = pd.DataFrame()    
for i in range(10):
    flag = df.sort_index(by=[i], ascending=[False])
    Ind = flag.index.values[:10]
    Words = [words[i] for i in Ind]
    topics[i] = pd.Series(Words)


# In[107]:

topics


# In[ ]:



