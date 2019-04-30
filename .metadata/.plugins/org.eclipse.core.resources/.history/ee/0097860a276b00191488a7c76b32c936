package org.apache.spark.run


import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{ MultivariateStatisticalSummary, Statistics }
import scala.collection.mutable.ListBuffer

import utils.keel.KeelParser

/**
 * @author Jesus Maillo
 */

object runBasicStatistics extends Serializable {

  var sc: SparkContext = null

  def main(arg: Array[String]) {

    if (arg.length < 3) {
      System.err.println("=> wrong parameters number")
      System.err.println("Parameters \n\t<path-to-train>\n\t<path-to-test>\n\t<path-to-output>")
      System.exit(1)
    }

    //Reading parameters
    val pathTrain = arg(0)
    val pathTest = arg(1)
    val pathOutput = arg(2)

    //Basic setup
    val jobName = "MLlib summary statictis"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    sc = new SparkContext(conf)

    //Reading the dataset
    val train = sc.textFile(pathTrain).map { line =>
      val array = line.split(",")
      var arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.cache

    val test = sc.textFile(pathTest).map { line =>
      val array = line.split(",")
      var arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.cache

    val observationsTrain = train.map(_.features)
    val observationsTest = test.map(_.features)

    // Compute column summary statistics. Param -> number of the column
    val summaryTrain: MultivariateStatisticalSummary = Statistics.colStats(observationsTrain)
    val summaryTest: MultivariateStatisticalSummary = Statistics.colStats(observationsTest)
    var outputString = new ListBuffer[String]
    outputString += "******TRAIN******\n\n"
    outputString += "@Max (0) --> " + summaryTrain.max(0) + "\n" // a dense vector containing the max value for each column
    outputString += "@Min (0) --> " + summaryTrain.min(0) + "\n"
    outputString += "@Mean (0) --> " + summaryTrain.mean(0) + "\n"
    outputString += "@Variance (0) --> " + summaryTrain.variance(0) + "\n" // columnwise variance
    outputString += "@NumNonZeros (0) --> " + summaryTrain.numNonzeros(0) + "\n" // number of nonzeros in each column
    outputString += "\n\n******TREST******\n\n"
    outputString += "@Max (0) --> " + summaryTest.max(0) + "\n" // a dense vector containing the max value for each column
    outputString += "@Min (0) --> " + summaryTest.min(0) + "\n"
    outputString += "@Mean (0) --> " + summaryTest.mean(0) + "\n"
    outputString += "@Variance (0) --> " + summaryTest.variance(0) + "\n" // columnwise variance
    outputString += "@NumNonZeros (0) --> " + summaryTest.numNonzeros(0) + "\n" // number of nonzeros in each column

    val predictionsTxt = sc.parallelize(outputString, 1)
    predictionsTxt.saveAsTextFile(pathOutput)
  }
}
