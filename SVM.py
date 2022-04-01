from ast import expr
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



if __name__ == "__main__":
    sc = SparkContext(appName="MNIST")
    spark = SparkSession(sc)
    Traindata = (spark.read.options(header = True, inferSchema = True).csv("s3://wittenbucket001/mnist_train.csv"))
    Testdata = (spark.read.options(header = True, inferSchema = True).csv("s3://wittenbucket001/mnist_test.csv"))
    # local test
    # Traindata = (spark.read.options(header = True, inferSchema = True).csv("/Users/witten/Downloads/assignment3/mnist-data/mnist_train.csv"))
    # Testdata = (spark.read.options(header = True, inferSchema = True).csv("/Users/witten/Downloads/assignment3/mnist-data/mnist_test.csv"))
    Feature = ["pixel" + str(i) for i in range(784)]
    vectorizer = VectorAssembler(inputCols=Feature, outputCol="feature")
    Train = (vectorizer.transform(Traindata).select("label", "feature").toDF("label", "feature").cache())
    Test = (vectorizer.transform(Testdata).select("label", "feature").toDF("label", "feature").cache())

    # Logistic Regression
    LGR = LogisticRegression(featuresCol="feature", labelCol="label", regParam=0.1, elasticNetParam=0.1, maxIter=1000)
    LGRM = LGR.fit(Train)
    TrainP = LGRM.transform(Train).withColumn("matched", expr("label == prediction"))
    EvaTrain = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    AccTrain = EvaTrain.evaluate(TrainP)
    print("LogisticRegression_Train_Accuracy:"+str(AccTrain))
    TestP = LGRM.transform(Test).withColumn("matched", expr("label == prediction"))
    TestP.show()
    Eva = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    Acc = Eva.evaluate(TestP)
    print("LogisticRegression_Test_Accuracy:"+str(Acc))

    # Random Forest
    RF = RandomForestClassifier(featuresCol="feature", labelCol="label",numTrees=10)
    RFM = RF.fit(Train)
    RFTrainP = RFM.transform(Train).withColumn("matched", expr("label == prediction"))
    RFEvaTrain = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    RFAccTrain = RFEvaTrain.evaluate(RFTrainP)
    print("RandomForest_Train_Accuracy:"+str(RFAccTrain))
    RFTestP = RFM.transform(Test).withColumn("matched", expr("label == prediction"))
    RFTestP.show()
    RFEva = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    RFAcc = RFEva.evaluate(RFTestP)
    print("RandomForest_Test_Accuracy:"+str(RFAcc))
