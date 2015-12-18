
# coding: utf-8

# In[2]:

import xml.etree.ElementTree as ET
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from datetime import datetime
import time
from time import mktime
import sys
from bs4 import BeautifulSoup
import sklearn
import sklearn.feature_extraction
import re
from bs4 import BeautifulSoup
from Word2VecUtility import Word2VecUtility


# In[9]:

from numpy import genfromtxt
aaId = genfromtxt('/Users/XW/Desktop/datascience.stackexchange.com/answerId.csv', delimiter=',')


# In[89]:

aaId = np.array(aaId).tolist()
aaId = [str(int(i)) for i in aaId]


# In[10]:

matrix = pd.read_csv('/Users/XW/Desktop/datascience.stackexchange.com/accepted.csv')
matrix1 = pd.read_csv('/Users/XW/Desktop/datascience.stackexchange.com/answer.csv')


# In[11]:

length = matrix.sum(axis=1)
length1 = matrix1.sum(axis = 1)


# In[12]:

post_tree=ET.parse('/Users/XW/Desktop/datascience.stackexchange.com/Posts.xml')
post=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot() if i.attrib.get("Id") in aaId]

post_frame=DataFrame(post,columns=['PostTypeId','CreationDate','Body'])
post_body=post_frame.loc[:,'Body']


# In[41]:

post1=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot() if i.attrib.get("PostTypeId") =='2' and i.attrib.get("Id") not in aaId]

post_frame1=DataFrame(post1,columns=['PostTypeId','CreationDate','Body'])
post_body1=post_frame1.loc[:,'Body']


# In[13]:

URL = []
for i in post_body:
    a = re.findall(r"/[a-zA-Z]*[:\/\/]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", i)
    URL.append(len(list(a)))
URL1 = []
for i in post_body1:
    a = re.findall(r"/[a-zA-Z]*[:\/\/]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", i)
    URL1.append(len(list(a)))


# In[14]:

#get answerer's reputation
QueId=[i.attrib.get("ParentId")  for i in post_tree.getroot() if i.attrib.get("Id") in aaId]
QueId1=[i.attrib.get("ParentId")  for i in post_tree.getroot() if i.attrib.get("PostTypeId") =='2' and i.attrib.get("Id") not in aaId]


# In[15]:

QCreatTime = [i.attrib.get("CreationDate")  for i in post_tree.getroot() if i.attrib.get("Id") in QueId]   


# In[16]:

QCreatTime1 = [[i.attrib.get("Id"),i.attrib.get("CreationDate")]  for i in post_tree.getroot() if i.attrib.get("Id") in QueId1] 


# In[17]:

ACreatTime = [i.attrib.get("CreationDate")  for i in post_tree.getroot() if i.attrib.get("Id") in aaId]  
ACreatTime1 = [[i.attrib.get("ParentId") ,i.attrib.get("CreationDate")]  for i in post_tree.getroot() if i.attrib.get("PostTypeId") =='2' and i.attrib.get("Id") not in aaId]  


# In[18]:

import datetime
Time = []
for i in range(len(ACreatTime)):
    gap = pd.to_datetime(ACreatTime[i]) - pd.to_datetime(QCreatTime[i])
    Time.append(gap)
Time1 = []
for i in ACreatTime1:
    for j in QCreatTime1:
        if i[0] == j[0]:

            gap = pd.to_datetime(i[1]) - pd.to_datetime(j[1])
            Time1.append(gap)
            break


# In[19]:

from datetime import timedelta
time = [i.total_seconds() for i in Time]
time1 = [i.total_seconds() for i in Time1]


# In[20]:

df2 = pd.DataFrame({ 'Train_Ind': np.array([1] * 309 + [0]*1262),
                     'Train_Time' : np.array(time[:309]+time1[:1262]),
                     'Train_len' : np.array(length[:309].tolist()+length1[:1262].tolist()), 
                     'Train_url'  : np.array(URL[:309] + URL1[:1262])})
train = df2.values


# In[21]:

df3 = pd.DataFrame({ 'Test_Time' : np.array(time[309:]+time1[1262:]),
                     'Test_len' : np.array(length[309:].tolist()+length1[1262:].tolist()), 
                     'Test_url'  : np.array(URL[309:] + URL1[1262:])
                   })
test = df3.values


# In[119]:

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train[0::,1::], train[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test)


# In[22]:

right =  [1] * 77 + [0]*316


# In[23]:

out = output.tolist()


# In[122]:

result = [int(i) for i in out]


# In[123]:

diff = [abs(right[i]-result[i]) for i in range(len(out))]


# In[124]:

sum(diff)/float(len(diff))


# In[26]:

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random


# In[195]:

false_positive_rate, true_positive_rate, thresholds = roc_curve(right, result)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[201]:

plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




# In[206]:

from sklearn import metrics
    # testing score
score = metrics.f1_score(right, result)
ACC = metrics.accuracy_score(right,result)


# In[207]:

score


# In[208]:

ACC


# In[210]:

output1 = forest.predict(train[0::,1::])
out1 = output1.tolist()


# In[222]:

right1 =  train[0::,0].tolist()
right2 = [int(i) for i in right1]


# In[223]:

result1 = [int(i) for i in out1]


# In[224]:

diff1 = [abs(right2[i]-result1[i]) for i in range(len(result1))]


# In[229]:

ACC = metrics.accuracy_score(right2,result1)


# In[230]:

#using logistic regression


# In[24]:

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# fit a logistic regression model to the data

logistic = LogisticRegression(C=0.1)
logistic.fit(train[0::,1::], train[0::,0])

print(logistic)
# make predictions
expected = right
predicted = logistic.predict(test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[27]:

false_positive_rate, true_positive_rate, thresholds = roc_curve(right, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[28]:

plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[132]:

from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC

model = SVC()
model.fit(train[0::,1::], train[0::,0])
print(model)
# make predictions
expected = right
predicted = model.predict(test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[126]:

train[0::,0]


# In[128]:

train[0::,1::]


# In[ ]:

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

