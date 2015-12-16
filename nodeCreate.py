import xml.etree.ElementTree as ET
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from datetime import datetime
import time
from time import mktime
import sys
sys.path.insert(0, '/Users/Zhen/Desktop/Courses/BigData/stackexchange/')
from Word2VecUtility import Word2VecUtility
import sklearn
import sklearn.feature_extraction
from bs4 import BeautifulSoup

post_tree=ET.parse('/Users/Zhen/Desktop/Courses/BigData/stackexchange/data/Posts.xml')
post_tree=ET.parse('/Users/Zhen/Desktop/Courses/BigData/stackexchange/Posts.xml')
post=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot()] 
post_frame=DataFrame(post,columns=['PostTypeId','CreationDate','Title','Body'])
a=[i.attrib.get("Tags").replace('>','').split('<') for i in post_tree.getroot() if (i.attrib.get("PostTypeId") in ['1','2'] & i.attrib.get("Tags") is not None)] 

###### psot node
post=[(i.attrib.get("Id"), i.attrib.get("PostTypeId"),\
		i.attrib.get("CreationDate"),\
		i.attrib.get("Title"),\
		BeautifulSoup(i.attrib.get("Body").replace('\n',''),'html.parser').get_text() ,\
		i.attrib.get("Tags").replace('>','').split('<')[1::2] if i.attrib.get("Tags") is not None else None,
		i.attrib.get("ViewCount"),\
		i.attrib.get("FavoriteCount"),
		'post')\
		 for i in post_tree.getroot() if i.attrib.get("PostTypeId") in ['1','2']]
post_frame=DataFrame(post,columns=['postID:ID','PostTypeId','CreationDate','Title','Body','Tags','ViewCount','FavoriteCount',':LABEL'])
post_frame.to_csv('post.csv',sep=';',encoding = 'utf-8',index=False)

####### 简易版
post=[(i.attrib.get("Id"), \
		i.attrib.get("CreationDate"),\
		i.attrib.get("Tags").replace('>','').split('<')[1::2] if i.attrib.get("Tags") is not None else None,
		i.attrib.get("ViewCount"),\
		i.attrib.get("FavoriteCount"),
		i.attrib.get("PostTypeId"))\
		 for i in post_tree.getroot() if i.attrib.get("PostTypeId") in ['1','2']]
post_frame=DataFrame(post,columns=['postID:ID','CreationDate','Tags','ViewCount','FavoriteCount',':LABEL'])
post_frame.to_csv('post2.csv',sep=';',encoding = 'utf-8',index=False)
##########

###### relationship
post_relation=[(i.attrib.get("Id"), 
		i.attrib.get("ParentId"),
		'post_relation')\
		 for i in post_tree.getroot() if i.attrib.get("PostTypeId") in ['2']]
post_relation_frame=DataFrame(post_relation,columns=[':START_ID',':END_ID',':TYPE'])
post_relation_frame.to_csv('post_relation.csv',sep=';',encoding = 'utf-8',index=False)

tag_tree=ET.parse('/Users/Zhen/Desktop/Courses/BigData/stackexchange/data/Tags.xml')
