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
      viewEvents = trainingData.viewEvents,
      rateEvents = trainingData.rateEvents,
      likeEvents = trainingData.likeEvents)
  }
}

class PreparedData(
                    val users: RDD[(String, User)],
                    val items: RDD[(String, Item)],
                    val viewEvents: RDD[ViewEvent],
                    val rateEvents: RDD[RateEvent],
                    val likeEvents: RDD[LikeEvent]
                  ) extends Serializable
