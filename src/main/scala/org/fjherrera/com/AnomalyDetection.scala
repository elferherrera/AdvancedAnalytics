package org.fjherrera.com
// https://github.com/sryza/aas.
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object AnomalyDetection extends App{
  Logger.getLogger("org").setLevel(Level.ERROR)

  val config = new SparkConf()
    .setMaster("local[*]")
    .setAppName("AnomalyDetection")

  val spark = SparkSession
    .builder
    .config(config)
    .getOrCreate()
  import spark.implicits._

  val file = args(0)
  val rawData = spark
    .read
    .option("inferSchema", true)
    .option("header", false)
    .csv(file)

  val data = rawData.toDF(
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label")

  data.select("label").groupBy("label").count().orderBy($"count".desc).show(10)

  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "_indexed")

    val encoder = new OneHotEncoder()
      .setInputCol(inputCol + "_indexed")
      .setOutputCol(inputCol + "_vec")

    val pipeline = new Pipeline().setStages(Array(indexer, encoder))

    (pipeline, inputCol + "_vec")
  }

  def clusteringScore(data: DataFrame, k: Int): Double = {
    println(s"Cluster training for $k")
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)

    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setWithStd(true)
      .setWithMean(false)
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")

    val kmeans = new KMeans()
      .setSeed(Random.nextLong())
      .setK(5)
      .setMaxIter(40)
      .setTol(1.0e-5)
      .setFeaturesCol("scaledFeatureVector")
      .setPredictionCol("cluster")

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)
    val predictions = pipelineModel.transform(data)
    val evaluator = new ClusteringEvaluator()
        .setFeaturesCol("scaledFeatureVector")
        .setPredictionCol("cluster")

    evaluator.evaluate(predictions)
  }

  (60 to 270 by 30).map(k => (k, clusteringScore(data, k))).foreach(println)

}
