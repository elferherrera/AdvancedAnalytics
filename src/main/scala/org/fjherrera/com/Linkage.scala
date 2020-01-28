package org.fjherrera.com

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}



object Linkage extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  // Spark context to run in spark cluster
  val sc = new SparkContext("local[*]", "Linkage")

  // Folder with all the data from the dir
  val dirFile = {
    if (args.length == 1) args(0)
    else  ""
  }

  // Raw information from rawBlocks
  val rawBlocks = sc.textFile(dirFile)

  // First line from the rawBlock
  println("First line from raw")
  println(rawBlocks.first)

  // Head from rawblock
  println("Sample from rawBlocks")
  rawBlocks.take(10).foreach(println)

  // Removing first line from raw info
  val filterLines = rawBlocks.filter(line => !line.contains("id_1"))

  // Lines without header
  println("filtered lines without header")
  filterLines.take(10).foreach(println)

  println("Creating SparkContext for RDD")
  val config: SparkConf = new SparkConf()
    .setMaster("local[*]")
    .setAppName("Linkage")

  val spark = SparkSession.builder.config(config).getOrCreate()
  import spark.implicits._

  val parsed = spark.read
    .option("header", "true")
    .option("nullValue", "?")
    .option("inferSchema", "true")
    .csv(dirFile)

  println(parsed.schema)

  // Preview dataframe
  parsed.show(5)

  // Preview number of rows
  println(parsed.count)

  // Stores in memory RDD to speed up future consults
  parsed.cache()

  println("Grouping row type and extracting is_match")
  println(parsed.rdd.map(row => row.getAs[Boolean]("is_match")).countByValue())

  println("Grouping using group")
  val groupMatch = parsed.groupBy("is_match")
    .count()
    .orderBy($"count".desc)
    .show()

  val summary = parsed.describe()
  summary.show()

  val schema = summary.schema

  val longForm = summary.flatMap(row => {
    val metric = row.getString(0)
    (1 until row.size).map(i => {
      (metric, schema(i).name, row.getString(i).toDouble)
    })
  })

  longForm.show()

  val longDF = longForm.toDF("metric", "field", "value")

  case class MatchData(
    id_1: Int,
    id_2: Int,
    cmp_fname_c1: Option[Double],
    cmp_fname_c2: Option[Double],
    cmp_lname_c1: Option[Double],
    cmp_lname_c2: Option[Double],
    cmp_sex: Option[Int],
    cmp_bd: Option[Int],
    cmp_bm: Option[Int],
    cmp_by: Option[Int],
    cmp_plz: Option[Int],
    is_match: Boolean
  )

  val matchData = parsed.as[MatchData]
  matchData.show(5)

  case class Score(value: Double) {
    def +(oi: Option[Int]) = {
      Score(value + oi.getOrElse(0))
    }
  }

  def scoreMatchData(md: MatchData): Double = {
    (Score(md.cmp_lname_c1.getOrElse((0.0))) + md.cmp_plz + md.cmp_by + md.cmp_bd + md.cmp_bm).value
  }

  val scored = matchData.map({
    md => (scoreMatchData(md), md.is_match)
  }).toDF("score", "is_match")

  scored.show()

  def crossTabs(scored: DataFrame, t:Double):DataFrame = {
    scored
      .selectExpr(s"score >= $t as above", "is_match")
      .groupBy("above")
      .pivot("is_match", Seq("true","false"))
      .count()
  }

  crossTabs(scored, 4.0).show()
  crossTabs(scored, 2.0).show()

}
