package com.martinywwan.supportvectormachine

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**Support Vector Machine implementation
  */
object SvmApp {

  Logger.getLogger("org").setLevel(Level.OFF) //remove debug logs
  Logger.getLogger("akka").setLevel(Level.OFF) //remove debug logs

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("DecisionTreeApplication").set("spark.driver.allowMultipleContexts", "true")
    val sc = new SparkContext(sparkConf);
    def supportVectorMachine = new SupportVectorMachine(sc);
    supportVectorMachine.generateModel("data/sample.txt")
  }
}
