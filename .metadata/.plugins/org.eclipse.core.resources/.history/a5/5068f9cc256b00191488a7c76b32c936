import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import utils.keel.KeelParser
import scala.collection.mutable.ListBuffer

// Read Header and dataset
val converter = new KeelParser(sc, "/home/spark/datasets/susy.header")

val train = sc.textFile("/home/spark/datasets/susy-10k-tra.data", 10).map(line => converter.parserToLabeledPoint(line)).cache
val test = sc.textFile("/home/spark/datasets/susy-10k-tst.data", 10).map(line => converter.parserToLabeledPoint(line)).cache

// Parameters
val k = 5
val dist = 2 //euclidean
val numClass = converter.getNumClassFromHeader()
val numFeatures = converter.getNumFeaturesFromHeader()
val numPartitionMap = 10
val numReduces = 2
val numIterations = 1
val maxWeight = 5

// Initialize the classifier
val knn = kNN_IS.setup(train, test, k, dist, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight)

// Classify
val predictions = knn.predict(sc)

// Obtaining the accuracy
val metrics = new MulticlassMetrics(predictions)
val precision = metrics.precision

// Obtaining the confusion matrix
val cm = metrics.confusionMatrix

// Obtaining the AUC
val binaryMetrics = new BinaryClassificationMetrics(predictions)
val AUC = binaryMetrics.areaUnderROC

// Showing results
println("\n Accuracy --> " + precision)
println("\n AUC --> " + AUC)
println("\n Confusion Matrix --> \n" + cm)
