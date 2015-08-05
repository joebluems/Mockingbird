name := "modeling"

version := "0.0.1"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
"org.apache.spark" %% "spark-core" % "1.3.1" % "provided",
"org.apache.spark" %% "spark-mllib" % "1.3.1"  % "provided",
"org.apache.lucene" % "lucene-analyzers-common" % "5.1.0"
)
