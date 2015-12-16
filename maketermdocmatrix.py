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

post_tree=ET.parse('/Users/Zhen/Desktop/Courses/BigData/stackexchange/data/Posts.xml')
post=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot()] 
post_frame=DataFrame(post,columns=['PostTypeId','CreationDate','Body'])
post_body=post_frame.loc[:,'Body']

clean_post = []
print "Cleaning and parsing the posts...\n"
for i in xrange( 0, len(post_body)):
		tmp=BeautifulSoup(post_body[i].rstrip('\n'),'html.parser').get_text()
		if tmp=='':
			continue
		f = file('parse/' + str(i), 'w')
		f.write(tmp.encode('utf-8'))


import textmining
import os
xDIR = 'parse'
def termdocumentmatrix_example(xDIR):
    
    # Initialize class to create term-document matrix
    count=0
    tdm = textmining.TermDocumentMatrix()
    for i in os.listdir(xDIR):
    	Res = tdm.add_doc(open(os.path.join(xDIR,i)).read()) 
    	if count>40: #如果是在所有文档上跑，就要把这里注释掉
    		break


    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    tdm.write_csv('matrix0.csv', cutoff=1) #输出结果
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
    for row in tdm.rows(cutoff=1):
            print row

termdocumentmatrix_example(xDIR)