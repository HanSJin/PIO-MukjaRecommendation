package org.template.ecommercerecommendation

import io.prediction.controller.PPreparator

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(
      users = trainingData.users,
      items = trainingData.items,
      rvEvents = trainingData.rvEvents,
      buyEvents = trainingData.buyEvents)
  }
}

class PreparedData(
  val users: RDD[(String, User)],
  val items: RDD[(String, Item)],
  val rvEvents: RDD[RVEvent],
  val buyEvents: RDD[BuyEvent]
) extends Serializable
