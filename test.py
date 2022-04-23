import findspark
findspark.init()

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.sql.types import *

schema = StructType().add("Label", IntegerType(), True).add("prediction", DoubleType(), True)

spark = SparkSession.builder.appName("Spark-Example").getOrCreate()

df_yog = spark.read.csv('../binary_data_6.csv', header=True, inferSchema=True, schema=schema)


df_yog.show()

df_yog.printSchema()
#df_yog = df_yog.na.fill(value=null)
'''
df = df_yog.head(15)
print(df)
for row in df:
	if row['prediction'] == 1.0:
		print(row)
print(df_yog.filter('Label = 1').count())
print(df_yog.filter('prediction = 1').count())

# Calculate the elements of the confusion matrix
TN = df_yog.filter('prediction = 0 AND Label = prediction').count()
TP = df_yog.filter('prediction = 1 AND Label = prediction').count()
FN = df_yog.filter('prediction = 0 AND Label <> prediction').count()
FP = df_yog.filter('prediction = 1 AND Label <> prediction').count()
# show confusion matrix
df_yog.groupBy('Label', 'prediction').count().show()

print(df_yog.count())
print(df_yog.filter('prediction < 0.5').count())
'''

df_yog = df_yog.na.drop()

# Calculate the elements of the confusion matrix
TN = df_yog.filter('prediction = 0 AND Label = prediction').count()
TP = df_yog.filter('prediction = 1 AND Label = prediction').count()
FN = df_yog.filter('prediction = 0 AND Label <> prediction').count()
FP = df_yog.filter('prediction = 1 AND Label <> prediction').count()
# show confusion matrix
df_yog.groupBy('Label', 'prediction').count().show()
expected = []
for row in df_yog.head(df_yog.count()):
 expected.append(row['Label'])
predicted = []
for row in df_yog.head(df_yog.count()):
 predicted.append(row['prediction'])
  
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(expected,predicted)

import matplotlib.pyplot as plt
import numpy as np
import itertools   
plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Lable')
tick_marks = np.arange(2) # length of classes
class_labels = ['Normal','AbNormal']
tick_marks
plt.xticks(tick_marks,class_labels)
plt.yticks(tick_marks,class_labels)
# plotting text value inside cells
thresh = cf.max() / 2.
for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],''),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')

plt.show()

accuracy = (TN + TP) / (TN + TP + FN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F =  2 * (precision*recall) / (precision + recall)

print('n precision: %0.3f' % precision)
print('n recall: %0.3f' % recall)
print('n accuracy: %0.3f' % accuracy)
print('n F1 score: %0.3f' % F)
