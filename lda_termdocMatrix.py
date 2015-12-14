from pyspark import SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext, Row
data = sc.textFile("/Users/Zhen/desktop/Courses/BigData/stackexchange/matrix0.csv")
header = data.first() #extract header
data = data.filter(lambda x:x !=header)
data = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(',')]))


# Index documents with unique IDs
corpus = data.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3)