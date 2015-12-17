import numpy as np
import pandas as pd

'''
preprocessing in command line
cat  /Users/Zhen/desktop/Courses/Bigdata/stackexchange/topicDist.txt/part-00000 /Users/Zhen/desktop/Courses/Bigdata/stackexchange/topicDist.txt/part-00001 > /Users/Zhen/desktop/Courses/Bigdata/stackexchange/topicsDict.txt

sed 's/(//g' topicsDict.txt >topicsDict0.txt

sed 's/)//g' topicsDict0.txt>topicsDict1.txt

sed 's/\[//g' topicsDict1.txt>topicsDict2.txt

sed 's/\]//g' topicsDict2.txt>topicsDict3.txt

'''
topics=pd.read_csv('/Users/Zhen/desktop/Courses/Bigdata/stackexchange/topicsDict3.txt', header=None)
topics.to_csv('/Users/Zhen/desktop/Courses/Bigdata/stackexchange/topicsDict3.csv', sep=',')