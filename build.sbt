name := "AdvancedAnalytics"

version := "0.1"

scalaVersion := "2.12.10"

val sparkVersion = "3.0.0-preview"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

libraryDependencies += "edu.umd" % "cloud9" % "1.4.9"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.5.2" artifacts (Artifact("stanford-corenlp", "models"), Artifact("stanford-corenlp"))

libraryDependencies += "info.bliki.wiki" % "bliki-core" % "3.0.19"
