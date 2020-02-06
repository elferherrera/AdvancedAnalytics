package org.fjherrera.com

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import scala.util.Random

object ForestCover extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  // Creating spark session with its config file to read
  // files using sql format
  val config: SparkConf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Recommender")
  val spark = SparkSession.builder.config(config).getOrCreate()
  import spark.implicits._

  // File name from directory
  val fileName = args(0)

  // Read CVS file as dataset
  val rawData = spark
    .read
    .option("inferSchema", true)
    .option("header", false)
    .csv(fileName)

  val colNames = Seq(
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
    ) ++ (
    (0 until 4).map(i => s"Wilderness_Area_$i")
    ) ++ (
    (0 until 40).map(i => s"Soil_Type_$i")
    ) ++ Seq("Cover_Type")

  // Creating a new DF with column names
  val data = rawData.toDF(colNames:_*)
    .withColumn("Cover_Type", $"Cover_Type".cast("double"))

  // Splitting the data for training and testing
  val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))

  // Cache data frames to speed calculations
  trainData.cache()
  testData.cache()

  // Creating vector inputs
  val inputCols = trainData.columns.filter(col => col != "Cover_Type")
  val assembler = new VectorAssembler()
    .setInputCols(inputCols)
    .setOutputCol("featureVector")

  val assembledTrainData = assembler.transform(trainData)
  assembledTrainData.select("featureVector").show(5)

  val classifier = new DecisionTreeClassifier()
    .setSeed(Random.nextLong())
    .setLabelCol("Cover_Type")
    .setFeaturesCol("featureVector")
    .setPredictionCol("prediction")

  val model = classifier.fit(assembledTrainData)

  println(model.toDebugString)
  model
    .featureImportances
    .toArray
    .zip(inputCols)
    .sorted
    .reverse
    .foreach(println)

  val predictions = model.transform(assembledTrainData)
  predictions.select("Cover_Type", "prediction", "probability").show(5)

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("Cover_Type")
    .setPredictionCol("prediction")

  val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
  val f1 = evaluator.setMetricName("f1").evaluate(predictions)

  println(s"Accuracy: $accuracy")
  println(s"f1: $f1")

  val predictionRDD = predictions
    .select("prediction", "Cover_Type")
    .as[(Double, Double)]
    .rdd

  val multiclassMetrics = new MulticlassMetrics(predictionRDD)
  print(multiclassMetrics.confusionMatrix)

  println("\nConfusion matrix with pivot table")
  val confusionMatrix = predictions
    .groupBy("Cover_Type")
    .pivot("prediction", (1 to 7))
    .count()
    .na
    .fill(0.0)
    .orderBy("Cover_Type")

  confusionMatrix.show()

  // Improving classification with different parameters
  println("Using a pipeline to train classifier")
  val assembler1 = new VectorAssembler()
    .setInputCols(inputCols)
    .setOutputCol("featureVector")

  val classifier1 = new DecisionTreeClassifier()
    .setSeed(Random.nextLong())
    .setLabelCol("Cover_Type")
    .setFeaturesCol("featureVector")
    .setPredictionCol("prediction")

  val pipeline = new Pipeline()
    .setStages(Array(assembler1, classifier1))

  val paramGrid = new ParamGridBuilder()
    .addGrid(classifier1.impurity, Seq("gini", "entropy"))
    .addGrid(classifier.maxDepth, Seq(1, 20))
    .addGrid(classifier.maxBins, Seq(40, 300))
    .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
    .build()

  val multiclassEval = new MulticlassClassificationEvaluator()
    .setLabelCol("Cover_Type")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val validator = new TrainValidationSplit()
    .setSeed(Random.nextLong())
    .setEstimator(pipeline)
    .setEvaluator(multiclassEval)
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.9)

  val validatorModel = validator.fit(trainData)

  val bestModel = validatorModel.bestModel
  print(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

  println("\n")
  println(validatorModel.validationMetrics.max)
  println(multiclassEval.evaluate(bestModel.transform(testData)))

}
