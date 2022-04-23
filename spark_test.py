import findspark
findspark.init()

import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import *

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers, regularizers
from keras.optimizers import adam_v2
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from elephas.ml_model import ElephasEstimator
from tensorflow.keras import optimizers

spark = SparkSession.builder.appName("Spark-Example").getOrCreate()

df_yog = spark.read.csv('Dataset-NF-UNSW-NB15-v2.csv', header=True, inferSchema=True)
df_yog.printSchema()


drop_col = ('IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'MIN_TTL','MAX_TTL')
df_yog = df_yog.drop(*drop_col)

df_yog, df_yog_du = df_yog.randomSplit([0.1, .9], seed=42)
df_yog, df_yog_2 = df_yog.randomSplit([0.1, .9], seed=42)
# Here, the last column is the category column
inputCols = df_yog.columns[0: len (df_yog.columns) - 2]

vecAssembler = VectorAssembler(\
                               inputCols = inputCols, \
                               outputCol = "va_features") \
                              .setHandleInvalid("skip")
                              
                              # Optional, as we are using Pipeline
vecTrain_data = vecAssembler.transform(df_yog)
vecTrain_data.select ("va_features").show(5, False)

from pyspark.ml.feature import StandardScaler
stdScaler = StandardScaler(inputCol="va_features", \
                        outputCol="features", \
                        withStd=True, \
                        withMean=False)
                        
# Optional - as we're using Pipeline
# Compute summary statistics by fitting the StandardScaler
scalerModel = stdScaler.fit(vecTrain_data)
# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(vecTrain_data)
scaledData.select ("features").show(5, False)

from pyspark.ml.feature import StringIndexer

#string_indexer = StringIndexer(inputCol="Attack", outputCol="Attack_index")
#indexer = string_indexer.fit(df_yog)


#pipeline = Pipeline(stages=[indexer, vecAssembler, stdScaler])
pipeline = Pipeline(stages=[vecAssembler, stdScaler])
pipelineModel = pipeline.fit(df_yog)

df_transform = pipelineModel.transform(df_yog)

df_transform.show()

#df_transform = df_transform.select('features', 'Attack_index')
df_transform = df_transform.select('features', 'Label')

training_data, testing_data = df_transform.randomSplit([.8, .2], seed=42)
# Number of classes
nb_classes = training_data.select('Label').distinct().count()

# Number of Inputs or Input Dimensions
input_dim = len(training_data.select('features').first()[0])

print(input_dim)

model = Sequential()
model.add(Dense(256, input_shape=(input_dim,), activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(256, activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

'''
model = Sequential()
model.add(Dense(512, input_shape=(input_dim,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
'''
model.summary()

adam = optimizers.Adam(lr=0.01)
opt_conf = optimizers.serialize(adam)

estimator = ElephasEstimator()
estimator.setFeaturesCol("features")
estimator.setLabelCol("Label")
estimator.set_keras_model_config(model.to_json())
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)
estimator.set_num_workers(1)
estimator.set_epochs(25) 
estimator.set_batch_size(128)
estimator.set_verbosity(1)
#estimator.set_verbosity(0)
estimator.set_validation_split(0.10)
estimator.set_optimizer_config(opt_conf)
estimator.set_mode("synchronous")
estimator.set_loss("binary_crossentropy")
estimator.set_metrics(['acc'])

# Create Deep Learning Pipeline
dl_pipeline = Pipeline(stages=[estimator])

fit_dl_pipeline = dl_pipeline.fit(training_data)
pred_train = fit_dl_pipeline.transform(training_data)
pred_test = fit_dl_pipeline.transform(testing_data)

pnl_train = pred_train.select('Label', "prediction")

#print(pnl_train.limit(10).toPandas())


pnl_train = pnl_train.withColumn("predictions", pnl_train["prediction"].getItem(1)).withColumn("value", pnl_train["prediction"].getItem(0))

pred_data = pnl_train.select('Label',round('predictions').alias('prediction'))
#pred_data, pred_data_2 = pred_data.randomSplit([0.1, .9], seed=42)
#print(pred_data.limit(50).toPandas())
#print(pred_data.filter('Label = 1').count())
#print(pred_data.filter('predictions > 0.5').count())
#pred_data.show(5000)
#pnl_train.select('Label','predictions').write.csv('DataAfterTrain')
#pred_data, pred_data_2 = pred_data.randomSplit([0.5, .5], seed=42)
pred_data.write.csv('binary_data_6.csv')

#pred_data_2.write.csv('pred_data.csv')

#pred_data = pnl_train.select('Label',round('predictions').alias('prediction'))
#pred_data.printSchema()

#pred_data = pred_data.select('Label', 'round(predictions, 0)')

#pred_data.write.csv('dataset_result.csv')


# Calculate the elements of the confusion matrix
# Calculate the elements of the confusion matrix
#TN = pred_data.filter('round(predictions, 0) = 0 AND Label = round(predictions, 0)').count()
#TP = pred_data.filter('round(predictions, 0) = 1 AND Label = round(predictions, 0)').count()
#FN = pred_data.filter('round(predictions, 0) = 0 AND Label <> round(predictions, 0)').count()
#FP = pred_data.filter('round(predictions, 0) = 1 AND Label <> round(predictions, 0)').count()
# show confusion matrix
#pred_data.groupBy('Label', 'round(predictions, 0)').count().show()
#print("Confusion Matrix", metrics.confusionMatrix())

