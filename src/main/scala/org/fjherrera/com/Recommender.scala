package org.fjherrera.com

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.apache.spark.ml.recommendation._

import scala.util.Random

object Recommender extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  // Creating spark session with its config file to read
  // files using sql format
  val config: SparkConf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Recommender")
  val spark = SparkSession.builder.config(config).getOrCreate()
  import spark.implicits._

  // Loading folder with data information from the arguments
  val dataDir = {
    if (args.length == 1) args(0)
    else  ""
  }

  // Raw data from user artist relationship
  val rawUserArtistData = spark.read.textFile(dataDir + "user_artist_data.txt")
  val userArtistDF = rawUserArtistData.map(line => {
    val Array(user, artist, _*) = line.split(" ")
    (user.toInt, artist.toInt)
  }).toDF("user", "artist")

  println("Max and min user and artist ID found in dataset")
  userArtistDF.agg(
    functions.min("user"),
    functions.max("user"),
    functions.min("artist"),
    functions.max("artist")).show()

  // Artist data from raw file
  val rawArtistData = spark.read.textFile(dataDir + "artist_data.txt")
  val artistByID = rawArtistData.flatMap(line => {
    val (id, name) = line.span(_ != '\t')
    if (name.isEmpty) {
      None
    } else {
      try {
        Some((id.toInt, name.trim))
      } catch {
        case _: NumberFormatException => None
      }
    }
  }).toDF("id", "name")

  // Alias information for artists
  val rawArtistAlias = spark.read.textFile(dataDir + "artist_alias.txt")
  val artistAlias = rawArtistAlias.flatMap(line => {
   val Array(artist, alias) = line.split('\t')
    if (artist.isEmpty) {
      None
    } else {
      Some((artist.toInt, alias.toInt))
    }
  }).collect().toMap
  println(artistAlias.head)

  artistByID.filter($"id" isin (1003929,1208690,1003926)).show()

  // Creating a broadcast variable with the name of the alias of
  // all the artists in the files
  val bArtistAlias = spark.sparkContext.broadcast(artistAlias)

  // Creating train data. Removing alias found in artist and creating
  // a cache variable which would speed up ALS calculations
  val trainData = rawUserArtistData.map(line => {
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      (userID, finalArtistID, count)
    }).toDF("user", "artist", "count")

  trainData.cache()

  // Creating ALS model for training and fitting data with users
  println("Training ALS model with artist user data")
  val model = new ALS()
    .setSeed(Random.nextLong())
    .setImplicitPrefs(true)
    .setRank(10)
    .setRegParam(0.01)
    .setAlpha(1.0)
    .setMaxIter(5)
    .setUserCol("user")
    .setItemCol("artist")
    .setRatingCol("count")
    .setPredictionCol("prediction")
    .fit(trainData)

  // Checking from recommendations for users
  val userID = 2093760
  val existingArtistID = trainData
    .filter($"user" === userID)
    .select("artist").as[Int].collect()

  artistByID.filter($"id" isin (existingArtistID:_*)).show()

  def makeRecommendations(
                         model: ALSModel,
                         userID: Int,
                         howMany: Int): DataFrame = {

    val toRecommend = model
      .itemFactors
      .select($"id".as("artist"))
      .withColumn("user", functions.lit(userID))

    model.transform(toRecommend)
      .select("artist", "prediction")
      .orderBy($"prediction".desc)
      .limit(howMany)
  }

  val topRecommendations = makeRecommendations(model, userID, 5)
  topRecommendations.show()

  val recommendedArtists = topRecommendations.select("artist").as[Int].collect()
  artistByID.filter($"id" isin (recommendedArtists:_*)).show()





}
