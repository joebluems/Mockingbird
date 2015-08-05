import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
 
object Models {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Mockingbird vs Watchman models")
    val sc = new SparkContext(conf)

    // read in data files (point to your location)
    val mock = sc.textFile("/Users/jblue/spark-1.3.1-bin-hadoop2.4/MOCKINGBIRD/mock.tokens")
    val watch = sc.textFile("/Users/jblue/spark-1.3.1-bin-hadoop2.4/MOCKINGBIRD/watch.tokens")

    // convert data to numeric features with TF 
    // we will consider Mockingbird passages = class 1, Watchman = class 0
    val tf = new HashingTF(10000)
    val mockData = mock.map { line =>
      var target = "1"
      LabeledPoint(target.toDouble, tf.transform(line.split(",")))
    }
    val watchData = watch.map { line =>
      var target = "0"
      LabeledPoint(target.toDouble, tf.transform(line.split(",")))
    }

    // build IDF model and transform data into modeling sets
    val data = mockData.union(watchData)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val trainDocs = splits(0).map{ x=>x.features}
    val idfModel = new IDF(minDocFreq = 3).fit(trainDocs)
    val train = splits(0).map{ point=>
      LabeledPoint(point.label,idfModel.transform(point.features))
    }
    train.cache()
    val test = splits(1).map{ point=>
      LabeledPoint(point.label,idfModel.transform(point.features))
    }

    println("Number of training examples: ", train.count())
    println("Number of test examples: ", test.count())

    // NAIVE BAYES MODEL
    println("Training Naive Bayes Model...")
    val nbmodel = NaiveBayes.train(train, lambda = 1.0)
    val bayesTrain = train.map(p => (nbmodel.predict(p.features), p.label))
    val bayesTest = test.map(p => (nbmodel.predict(p.features), p.label))
    println("NB Training accuracy: ",bayesTrain.filter(x => x._1 == x._2).count() / bayesTrain.count().toDouble)
    println("NB Test set accuracy: ",bayesTest.filter(x => x._1 == x._2).count() / bayesTest.count().toDouble)
    println("Naive Bayes Confusion Matrix:")
    println("Predict:mock,label:mock -> ",bayesTest.filter(x => x._1 == 1.0 & x._2==1.0).count())
    println("Predict:watch,label:watch -> ",bayesTest.filter(x => x._1 == 0.0 & x._2==0.0).count())
    println("Predict:mock,label:watch -> ",bayesTest.filter(x => x._1 == 1.0 & x._2==0.0).count())
    println("Predict:watch,label:mock -> ",bayesTest.filter(x => x._1 == 0.0 & x._2==1.0).count())

    // RANDOM FOREST MODEL
    println("Training Random Forest Regression Model...")
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numClasses = 2
    val featureSubsetStrategy = "auto" 
    val impurity = "variance"
    val maxDepth = 10
    val maxBins = 32
    val numTrees = 50 
    val modelRF = RandomForest.trainRegressor(train, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Calculating random forest metrics 
    val trainScores = train.map { point =>
      val prediction = modelRF.predict(point.features)
      (prediction, point.label)
    }
    val testScores = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (prediction, point.label)
    }
    val metricsTrain = new BinaryClassificationMetrics(trainScores,100)
    val metricsTest = new BinaryClassificationMetrics(testScores,100)
    println("RF Training AuROC: ",metricsTrain.areaUnderROC())
    println("RF Test AuROC: ",metricsTest.areaUnderROC())

    // writing out ROC - don't forget to change the location of the ROC
    println("Attempting to write RF ROC to file...")
    val trainroc= metricsTrain.roc()
    val testroc= metricsTest.roc()
    trainroc.saveAsTextFile("/Users/jblue/MOCKINGBIRD/ROC/rftrain")
    testroc.saveAsTextFile("/Users/jblue/MOCKINGBIRD/ROC/rftest")

    // GRADIENT BOOSTED TREES REGRESSION MODEL
    println("Training Gradient Boosted Trees Regression Model...")
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 50 // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val modelGB = GradientBoostedTrees.train(train, boostingStrategy)

    // Calculating gradient boosted metrics 
    val trainScores2 = train.map { point =>
      val prediction = modelGB.predict(point.features)
      (prediction, point.label)
    }
    val testScores2 = test.map { point =>
      val prediction = modelGB.predict(point.features)
      (prediction, point.label)
    }
    val metricsTrain2 = new BinaryClassificationMetrics(trainScores2,100)
    val metricsTest2 = new BinaryClassificationMetrics(testScores2,100)
    println("GB Training AuROC: ",metricsTrain2.areaUnderROC())
    println("GB Test AuROC: ",metricsTest2.areaUnderROC())

    // writing out ROC - don't forget to change the location of the ROC
    println("Attempting to write GB ROC to file...")
    val trainroc2= metricsTrain2.roc()
    val testroc2= metricsTest2.roc()
    trainroc2.saveAsTextFile("/Users/jblue/MOCKINGBIRD/ROC/gbtrain")
    testroc2.saveAsTextFile("/Users/jblue/MOCKINGBIRD/ROC/gbtest")


  }
}
