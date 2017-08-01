package com.martinywwan.supportvectormachine

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

/**Support Vector Machine implementation
  *Information about libsvm data file reader: https://github.com/cjlin1/libsvm/blob/master/README
  *
  ** @param sc - SparkContext (standalone application)
  */
class SupportVectorMachine(sc : SparkContext) {

  def generateModel(dataFile : String) : Unit =  {
    val data = MLUtils.loadLibSVMFile(sc, dataFile) // Load and parse the data file (LibSVM file format)

    // Split the RDD data into training and test sets
    val splits = data.randomSplit(Array(0.7, 0.3)) //label - feature pairs
    val trainingData= splits(0)
    val testData = splits(1)

    val iterations = 50
    val model = SVMWithSGD.train(trainingData, iterations) //Stochastic Gradient Descent
    model.clearThreshold()

    // Evaluate model on test instances
    val predAndLabel = testData.map { point =>
      val prediction = model.predict(point.features)
      (prediction , point.label)
    }

    val metrics = new BinaryClassificationMetrics(predAndLabel)
    val areaUnderROC = metrics.areaUnderROC() //https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it

    println("Area under the Curver = " + areaUnderROC) //1 = perfect prediction model, AUROC<0.5 = re-evaluate the model
  }
}
