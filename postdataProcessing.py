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

post_tree=ET.parse('/Users/Zhen/Desktop/Courses/BigData/stackexchange/Posts.xml')
post=[(i.attrib.get("PostTypeId"),i.attrib.get("CreationDate"),i.attrib.get("Body") ) for i in post_tree.getroot()] 
post_frame=DataFrame(post,columns=['PostTypeId','CreationDate','Body'])
post_body=post_frame.loc[:,'Body']

clean_post = []
print "Cleaning and parsing the posts...\n"
for i in xrange( 0, len(post_body)):
	clean_post.append(" ".join(Word2VecUtility.review_to_wordlist(post_body[i], True)))
clean_postdf=pd.DataFrame(clean_post)
clean_postdf.to_csv('post_body.csv',sep=',',encoding = 'utf-8')


 # Initialize the "CountVectorizer" object, which is scikit-learn's
 # bag of words tool.
vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000,min_df=1)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
data_features = vectorizer.fit_transform(clean_post)

# Numpy arrays are easy to work with, so convert the result to an
# array
data_features = data_features.toarray()

print('data_features: {0}'.format(data_features))
print('vectorizer.vocabulary_: {0}'.format(vectorizer.vocabulary_))



from pyspark import SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

# Load and parse the data
sc =SparkContext()
corpus = sc.parallelize(data_features)























'''
import textmining
def termdocumentmatrix_example(textlist):
    # Create some very short sample documents

    # Initialize class to create term-document matrix
	tdm = textmining.TermDocumentMatrix()
    # Add the documents
	for i in xrange(0,len(textlist)):
		tdm.add_doc(textlist[i])

    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
	tdm.write_csv('matrix.csv', cutoff=1)
    # Instead of writing out the matrix you can also access its rows directly.
    # Let's print them to the screen.
	for row in tdm.rows(cutoff=1):
			print row


termdocumentmatrix_example(clean_post)




import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 

def fn_tdm_df(docs, xColNames = None, **kwargs):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''

    #initialize the  vectorizer
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    #create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names())
    if xColNames is not None:
        df.columns = xColNames

    return df

DIR = 'C:/Data/'

def fn_CorpusFromDIR(xDIR):
    ''' functions to create corpus from a Directories
    Input: Directory
    Output: A dictionary with 
             Names of files ['ColNames']
             the text in corpus ['docs']'''
    import os
    Res = dict(docs = [open(os.path.join(xDIR,f)).read() for f in os.listdir(xDIR)],
               ColNames = map(lambda x: 'P_' + x[0:6], os.listdir(xDIR)))
    return Res

d1 = fn_tdm_df(docs = fn_CorpusFromDIR(DIR)['docs'],
          xColNames = fn_CorpusFromDIR(DIR)['ColNames'], 
          stop_words=None, charset_error = 'replace')  