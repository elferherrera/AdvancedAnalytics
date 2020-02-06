package org.fjherrera.com

import org.apache.log4j.{Level, Logger}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io._
import org.apache.spark.SparkConf
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{CountVectorizer, IDF}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import java.util.Properties

import edu.umd.cloud9.collection.XMLInputFormat
import edu.umd.cloud9.collection.wikipedia.language._
import edu.umd.cloud9.collection.wikipedia._
import edu.stanford.nlp.pipeline._
import edu.stanford.nlp.ling.CoreAnnotations._


object LatentSentiment extends App {
  // Function to parse the XML dumps using Cloud9 project
  def wikiXmlToPlainText(pageXml: String): Option[(String, String)] = {
    val hackedPageXml = pageXml.replaceFirst(
      "<text xml:space=\"preserve\" bytes=\"\\d+\">",
      "<text xml:space=\"preserve\">",
    )

    val page = new EnglishWikipediaPage()
    WikipediaPage.readPage(page, hackedPageXml)
    if (page.isEmpty) None
    else Some((page.getTitle, page.getContent))
  }

  // NLP pipeline
  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  def plainTextToLemmas(text: String,
                        stopWords: Set[String],
                        pipeline: StanfordCoreNLP): Seq[String] = {

    val doc = new Annotation(text)
    pipeline.annotate(doc)

    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala;
         token <- sentence.get(classOf[TokensAnnotation]).asScala){

      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma)
        && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }

  Logger.getLogger("org").setLevel(Level.ERROR)

  val config = new SparkConf()
    .setMaster("local[*]")
    .setAppName("LatentSentiment")

  val spark = SparkSession
    .builder
    .config(config)
    .getOrCreate()
  import spark.implicits._

  // File with wikipedia dump
  val fileWikipedia = args(0)
  val fileStopWords = args(1)

  @transient val conf = new Configuration()
  conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
  conf.set(XMLInputFormat.END_TAG_KEY, "</page>")

  val kvs = spark
    .sparkContext
    .newAPIHadoopFile(fileWikipedia, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf)

  // Creating raw DS with raw data from XML
  val rawXmls = kvs.map(_._2.toString).toDS()
  val docTexts = rawXmls.filter(_ != null).flatMap(wikiXmlToPlainText)

  // Stop words to use for filtering the documents
  val stopWords = scala.io.Source.fromFile(fileStopWords).getLines().toSet
  val bStopWords = spark.sparkContext.broadcast(stopWords)

  val terms: Dataset[(String, Seq[String])] =
    docTexts.mapPartitions({
      iter => {
        val pipeline = createNLPPipeline()
        iter.map({
          case(title, contents) =>
            (title, plainTextToLemmas(contents, bStopWords.value, pipeline))
        })
      }
    })

  val termsDF = terms.toDF("title", "terms")
  val filtered = termsDF.where(size($"terms") > 1)
  val numTerms = 20000
  val countVectorizer = new CountVectorizer()
    .setInputCol("terms").setOutputCol("termFreqs")
    .setVocabSize(numTerms)

  val vocabModel = countVectorizer.fit(filtered)
  val docTermFreqs = vocabModel.transform(filtered)

  docTermFreqs.cache()

  val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidVec")
  val idfModel = idf.fit(docTermFreqs)
  val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidVec")

  val termIds: Array[String] = vocabModel.vocabulary

  val docIds = docTermFreqs.rdd.map(_.getString(0))
    .zipWithUniqueId()
    .map(_.swap)
    .collect()
    .toMap

  docTermFreqs.show(5)

}
