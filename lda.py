from pyspark import SparkContext
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext, Row
# Load and parse the data
sc =SparkContext()
data = sc.textFile("/Users/Zhen/desktop/Courses/BigData/stackexchange/post_body.csv")
parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
# textFile = sc.textFile("/Users/Zhen/desktop/Courses/BigData/stackexchange/post_body.csv").map(lambda line : line.split(' '))


# Index documents with unique IDs
corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3)

# Output topics. Each is a distribution over words (matching word count vectors)
print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
topics = ldaModel.topicsMatrix()
for topic in range(3):
    print("Topic " + str(topic) + ":")
    for word in range(0, ldaModel.vocabSize()):
        print(" " + str(topics[word][topic]))
		
# Save and load model
ldaModel.save(sc, "/Users/Zhen/desktop/Courses/BigData/stackexchange/lda_result")
sameModel = LDAModel.load(sc, "/Users/Zhen/desktop/Courses/BigData/stackexchange/lda_result")