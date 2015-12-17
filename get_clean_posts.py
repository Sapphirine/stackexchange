
# coding: utf-8

# In[37]:

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


# In[88]:

from numpy import genfromtxt
aaId = genfromtxt('/Users/XW/Desktop/datascience.stackexchange.com/answerId.csv', delimiter=',')


# In[89]:

aaId = np.array(aaId).tolist()
aaId = [str(int(i)) for i in aaId]


# In[90]:

post_tree=ET.parse('/Users/XW/Desktop/datascience.stackexchange.com/Posts.xml')
post=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot() if i.attrib.get("Id") in aaId]

post_frame=DataFrame(post,columns=['PostTypeId','CreationDate','Body'])
post_body=post_frame.loc[:,'Body']


# In[41]:

post1=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot() if i.attrib.get("PostTypeId") =='2' and i.attrib.get("Id") not in aaId]

post_frame1=DataFrame(post1,columns=['PostTypeId','CreationDate','Body'])
post_body1=post_frame1.loc[:,'Body']


# In[42]:

clean_post = []
for i in xrange( 0, len(post_body1)):
    tmp=BeautifulSoup(post_body1[i].rstrip('\n'),'html.parser').get_text()
    if tmp=='':
        continue
        clean_post=" ".join(Word2VecUtility.review_to_wordlist(tmp, True))
    f = file('/Users/XW/Desktop/datascience.stackexchange.com/parse0/' + str(i), 'w')
    f.write(tmp.encode('utf-8'))


# In[43]:

import textmining
import os
xDIR = '/Users/XW/Desktop/datascience.stackexchange.com/parse0'
def termdocumentmatrix_example(xDIR):
    
    # Initialize class to create term-document matrix
    count=0
    tdm = textmining.TermDocumentMatrix()
    for i in os.listdir(xDIR):
    	Res = tdm.add_doc(open(os.path.join(xDIR,i)).read()) 
termdocumentmatrix_example(xDIR)


# In[95]:

clean_post = []
for i in xrange( 0, len(post_body)):
    tmp=BeautifulSoup(post_body[i].replace('\n',""),'html.parser').get_text()
    if tmp=='':
        continue
    clean_post=" ".join(Word2VecUtility.review_to_wordlist(tmp, True))
    f = file('/Users/XW/Desktop/datascience.stackexchange.com/parse/' + str(i), 'w')
    f.write(clean_post.encode('utf-8'))


# In[97]:

import textmining
import os
xDIR = '/Users/XW/Desktop/datascience.stackexchange.com/parse'
def termdocumentmatrix_example(xDIR):
    
    # Initialize class to create term-document matrix
    count=0
    tdm = textmining.TermDocumentMatrix()
    for i in os.listdir(xDIR):
        Res = tdm.add_doc(open(os.path.join(xDIR,i)).read()) 


    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    tdm.write_csv('/Users/XW/Desktop/datascience.stackexchange.com/accepted.csv',cutoff=1) #输出结果
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.


termdocumentmatrix_example(xDIR)


