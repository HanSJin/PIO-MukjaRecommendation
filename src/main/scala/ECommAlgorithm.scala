package org.template.ecommercerecommendation

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import io.prediction.data.storage.BiMap
import io.prediction.data.storage.Event
import io.prediction.data.store.LEventStore

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.{Rating => MLlibRating}
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger

import scala.collection.mutable.PriorityQueue
import scala.concurrent.duration.Duration
import scala.concurrent.ExecutionContext.Implicits.global

case class ECommAlgorithmParams(
  appName: String,
  unseenOnly: Boolean,
  seenEvents: List[String],
  similarEvents: List[String],
  rank: Int,
  numIterations: Int,
  lambda: Double,
  seed: Option[Long]
) extends Params


case class ProductModel(
  item: Item,
  features: Option[Array[Double]], // features by ALS
  count: Double // popular count for default score
)

class ECommModel(
  val rank: Int,
  val userFeatures: Map[Int, Array[Double]],
  val productModels: Map[Int, ProductModel],
  val userStringIntMap: BiMap[String, Int],
  val itemStringIntMap: BiMap[String, Int]
) extends Serializable {

  @transient lazy val itemIntStringMap = itemStringIntMap.inverse

  override def toString = {
    s" rank: ${rank}" +
    s" userFeatures: [${userFeatures.size}]" +
    s"(${userFeatures.take(2).toList}...)" +
    s" productModels: [${productModels.size}]" +
    s"(${productModels.take(2).toList}...)" +
    s" userStringIntMap: [${userStringIntMap.size}]" +
    s"(${userStringIntMap.take(2).toString}...)]" +
    s" itemStringIntMap: [${itemStringIntMap.size}]" +
    s"(${itemStringIntMap.take(2).toString}...)]"
  }
}

class ECommAlgorithm(val ap: ECommAlgorithmParams)
  extends P2LAlgorithm[PreparedData, ECommModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): ECommModel = {
    require(!data.viewEvents.take(1).isEmpty,
      s"viewEvents in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    require(!data.rateEvents.take(1).isEmpty,
      s"rateEvents in PreparedData cannot be empty." +
        " Please check if DataSource generates TrainingData" +
        " and Preprator generates PreparedData correctly.")
    require(!data.likeEvents.take(1).isEmpty,
      s"likeEvents in PreparedData cannot be empty." +
        " Please check if DataSource generates TrainingData" +
        " and Preprator generates PreparedData correctly.")
    require(!data.users.take(1).isEmpty,
      s"users in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    require(!data.items.take(1).isEmpty,
      s"items in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
    // create User and item's String ID to integer index BiMap
    val userStringIntMap = BiMap.stringInt(data.users.keys)
    val itemStringIntMap = BiMap.stringInt(data.items.keys)

    val mllibRatings: RDD[MLlibRating] = genMLlibRating(
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      data = data
    )

    // MLLib ALS cannot handle empty training data.
    require(!mllibRatings.take(1).isEmpty,
      s"mllibRatings cannot be empty." +
      " Please check if your events contain valid user and item ID.")

    // seed for MLlib ALS
    val seed = ap.seed.getOrElse(System.nanoTime)

    // use ALS to train feature vectors
    val m = ALS.trainImplicit(
      ratings = mllibRatings,
      rank = ap.rank,
      iterations = ap.numIterations,
      lambda = ap.lambda,
      blocks = -1,
      alpha = 1.0,
      seed = seed)

    val userFeatures = m.userFeatures.collectAsMap.toMap

    // convert ID to Int index
    val items = data.items.map { case (id, item) =>
      (itemStringIntMap(id), item)
    }

    // join item with the trained productFeatures
    val productFeatures: Map[Int, (Item, Option[Array[Double]])] =
      items.leftOuterJoin(m.productFeatures).collectAsMap.toMap

    val popularCount = trainDefault(
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap,
      data = data
    )

    val productModels: Map[Int, ProductModel] = productFeatures
      .map { case (index, (item, features)) =>
        val pm = ProductModel(
          item = item,
          features = features,
          // NOTE: use getOrElse because popularCount may not contain all items.
          count = popularCount.getOrElse(index, 0.0)
        )
        (index, pm)
      }

    new ECommModel(
      rank = m.rank,
      userFeatures = userFeatures,
      productModels = productModels,
      userStringIntMap = userStringIntMap,
      itemStringIntMap = itemStringIntMap
    )
  }

  /** Generate MLlibRating from PreparedData.
    * You may customize this function if use different events or different aggregation method
    */
  def genMLlibRating(
    userStringIntMap: BiMap[String, Int],
    itemStringIntMap: BiMap[String, Int],
    data: PreparedData): RDD[MLlibRating] = {

    val v_mllibRatings = data.viewEvents
      .map { r =>
        // Convert user and item String IDs to Int index for MLlib
        //입력 값 제대로 들어왔나 확인 후 이상있는 item이나 user값을 -1로 변경한다
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), 1.0)
      }
      .filter { case ((u, i), v) =>
        // item이나 user값이 -1인 값들을 제외시킨다
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .reduceByKey(_ + _) // aggregate all view events of same user-item pair //키가 같은 것 끼리 v값을 더함
      .map { case ((u, i), v) =>
      // MLlibRating requires integer index for user and item
      //최종 값들 저장
      ((u, i), v)
    }
      .cache()

    val r_mllibRatings = data.rateEvents
      .map { r =>
        // Convert user and item String IDs to Int index for MLlib
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), (r.rating, r.t))
      }
      .filter { case ((u, i), v) =>
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .reduceByKey { case (v1, v2) => // MODIFIED
        // if a user may rate same item with different value at different times,
        // use the latest value for this case.
        // Can remove this reduceByKey() if no need to support this case.
        val (rating1, t1) = v1
        val (rating2, t2) = v2
        // keep the latest value
        if (t1 > t2) v1 else v2
      }
      .map { case ((u, i), (rating,t)) => // MODIFIED
        // MLlibRating requires integer index for user and item
        ((u, i), rating/2.5) // MODIFIED
      }.cache()

    val l_mllibRatings = data.likeEvents
      .map { r =>
        // Convert user and item String IDs to Int index for MLlib
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), (r.like, r.t))
      }.filter { case ((u, i), v) =>
      (u != -1) && (i != -1)
      }.reduceByKey { case (v1, v2) => // MODIFIED
        val (like1, t1) = v1
        val (like2, t2) = v2
        if (t1 > t2) v1 else v2
      }.map { case ((u, i), (like, t)) => // MODIFIED
        val r = if (like) 5.0 else 0.0
        ((u, i), r)
      }.cache()

    val sum_mllibRatings = v_mllibRatings.union(l_mllibRatings).reduceByKey(_ + _).cache()

    val total_mllibRatings = sum_mllibRatings.union(r_mllibRatings).reduceByKey(_ * _)
      .map { case ((u, i), v) =>
        // MLlibRating은 user와 item에 대한 integer index를 필요로 함
        MLlibRating(u, i, v)
      }.cache()


    total_mllibRatings
  }

  /** Train default model.
    * You may customize this function if use different events or
    * need different ways to count "popular" score or return default score for item.
    */
  def trainDefault(
    userStringIntMap: BiMap[String, Int],
    itemStringIntMap: BiMap[String, Int],
    data: PreparedData): Map[Int, Double] = {
    // count number of likes
    // (item index, count)
    val viewCountsRDD: RDD[(Int, Double)] = data.viewEvents
      .map { r =>
        // Convert user and item String IDs to Int index
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        (uindex, iindex, 1)
      }
      .filter { case (u, i, v) =>
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .map { case (u, i, v) => (i, 1.0) } // key is item
      .reduceByKey(_ + _) // count number of items occurrence

    val rateCountsRDD: RDD[(Int, Double)] = data.rateEvents
      .map { r =>
        // Convert user and item String IDs to Int index
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), (r.rating, r.t))
      }
      .filter { case ((u, i), v) =>
        // keep events with valid user and item index
        (u != -1) && (i != -1)
      }
      .reduceByKey { case (v1, v2) => // MODIFIED
        val (rating1, t1) = v1
        val (rating2, t2) = v2
        // keep the latest value
        if (t1 > t2) v1 else v2
      }
      .map { case ((u, i), (rating,t)) => (i, rating/2.5) } // key is item
      .reduceByKey(_ + _) // count number of items occurrence

    val likeCountsRDD: RDD[(Int, Double)] = data.likeEvents
      .map { r =>
        // Convert user and item String IDs to Int index
        val uindex = userStringIntMap.getOrElse(r.user, -1)
        val iindex = itemStringIntMap.getOrElse(r.item, -1)

        if (uindex == -1)
          logger.info(s"Couldn't convert nonexistent user ID ${r.user}"
            + " to Int index.")

        if (iindex == -1)
          logger.info(s"Couldn't convert nonexistent item ID ${r.item}"
            + " to Int index.")

        ((uindex, iindex), (r.like, r.t))
      }
      .filter { case ((u, i), v) =>
        (u != -1) && (i != -1)
      }
      .reduceByKey { case (v1, v2) => // MODIFIED
        val (like1, t1) = v1
        val (like2, t2) = v2
        if (t1 > t2) v1 else v2
      }
      .map { case ((u, i), (like, t)) => // MODIFIED
        val r = if (like) 5.0 else 0.0
        (i, r)
      }
      .reduceByKey(_ + _)

    val sum_eventsRDD: RDD[(Int, Double)] = viewCountsRDD.union(likeCountsRDD).reduceByKey(_ + _)

    val resultRDD: RDD[(Int, Double)] = sum_eventsRDD.union(rateCountsRDD).reduceByKey(_ * _)

    resultRDD.collectAsMap.toMap
  }

  def predict(model: ECommModel, query: Query): PredictedResult = {
	  logger.info(s"userFeature : ${query.users.mkString(",")}.")

    var algo = 4;

    val userFeatures = model.userFeatures
    val productModels = model.productModels

    // convert whiteList's string ID to integer index
    val whiteList: Option[Set[Int]] = query.whiteList.map( set =>
      set.flatMap(model.itemStringIntMap.get(_))
    )

    val finalBlackList: Set[Int] = genBlackList(query = query)
      // convert seen Items list from String ID to interger Index
      .flatMap(x => model.itemStringIntMap.get(x))

    val topScores: Array[(Int, Double)] =  if (query.items.isDefined) {
      //logger.info(s"similar ${query.items}.")
      print(s"******similar 영역 안에 들어옴 ******")

      //recentItems과 동일한 버전
      val queryList: Set[Int] = query.items.get.flatMap(x => model.itemStringIntMap.get(x))

      val queryFeatures: Vector[Array[Double]] = queryList.toVector
        // productModels may not contain the requested item
        .map { i =>
        productModels.get(i).flatMap { pm => pm.features }
      }.flatten


      if (queryFeatures.isEmpty) {
        logger.info(s"Similar / Fail : ${query.items}.")
        //logger.info(s"No productFeatures vector for query items ${query.items}.")
        print(s"******${query.items}. productFeatures X 없어요 ******")
        Array[(Int, Double)]()
      } else {
        logger.info(s"Similar / Success : ${query.items}.")
        print(s"******${query.items}. productFeatures O 있어요 ******")
        predictSimilar(
          queryList = queryList,
          queryFeatures = queryFeatures,
          productModels = productModels,
          query = query,
          whiteList = whiteList,
          blackList = finalBlackList
        )
      }
    }
    else {
      val userFeature: Option[Array[Double]] =
        model.userStringIntMap.get(query.users(0)).flatMap { userIndex =>
          userFeatures.get(userIndex)
        }
      print(s"recommand.")
      //logger.info(s"Recommand / KnownUser items : ${recentItems}.")

      if (userFeature.isDefined && query.users.length==1) {
        algo = 1;
        // the user has feature vector
        predictKnownUser(
          userFeature = userFeature.get,
          productModels = productModels,
          query = query,
          whiteList = whiteList,
          blackList = finalBlackList
        )
      } else
      {
          // the user doesn't have feature vector.
          // For example, new user is created after model is trained.
          //logger.info(s"No userFeature found for user ${query.users.mkString(",")}.")
          logger.info(s"No userFeature found for user ${query.users(0)}.")

          // check if the user has recent events on some items
          val recentItems: Set[String] = getRecentItems(query)
          val recentList: Set[Int] = recentItems.flatMap(x =>
            model.itemStringIntMap.get(x))

          val recentFeatures: Vector[Array[Double]] = recentList.toVector
            // productModels may not contain the requested item
            .map { i =>
            productModels.get(i).flatMap { pm => pm.features }
          }.flatten

          if (recentFeatures.isEmpty) {
            algo = 2;
            //logger.info(s"No features vector for recent items ${recentItems}.")
            logger.info(s"Recommand / Popular items : ${recentItems}.")
            predictDefault(
              productModels = productModels,
              query = query,
              whiteList = whiteList,
              blackList = finalBlackList
            )
          } else {
            algo = 3;
            logger.info(s"Recommand / Similar items : ${recentItems}.")
            predictSimilarRecommendation(
              recentFeatures = recentFeatures,
              productModels = productModels,
              query = query,
              whiteList = whiteList,
              blackList = finalBlackList
            )
          }
        }
      }

    val itemScores = topScores.map { case (i, s) =>
      new ItemScore(
        // convert item int index back to string ID
        algo = algo,
        item = model.itemIntStringMap(i),
        score = s
      )
    }

    new PredictedResult(itemScores)
  }


  /** 변경했음 오류나거나 작동 제대로 안하면 e-commerce blacklist가져오기*/
  /** Generate final blackList based on other constraints */
  def genBlackList(query: Query): Set[String] = {

    // if unseenOnly is True, get all seen items
    val seenItems: Set[String] = if (ap.unseenOnly) {
      var seenItems1:Set[String] = Set(null)
      for (i <- 0 to query.users.size-1) {

        // get all user item events which are considered as "seen" events
        val seenEvents: Iterator[Event] = try {
          LEventStore.findByEntity(
            appName = ap.appName,
            entityType = "user",
            entityId = query.users(i),
            eventNames = Some(ap.seenEvents),
            targetEntityType = Some(Some("item")),
            // set time limit to avoid super long DB access
            timeout = Duration(200, "millis")
          )
        } catch {
          case e: scala.concurrent.TimeoutException =>
            logger.error(s"Timeout when read seen events." +
              s" Empty list is used. ${e}")
            Iterator[Event]()
          case e: Exception =>
            logger.error(s"Error when read seen events: ${e}")
            throw e
        }

        val seenItems2:Set[String] = seenEvents.map { event =>
          try {
            event.targetEntityId.get
          } catch {
            case e => {
              logger.error(s"Can't get targetEntityId of event ${event}.")
              throw e
            }
          }
        }.toSet

        if(i==0)
          seenItems1 = seenItems2
        else
          seenItems1.intersect(seenItems2).toSet

      }
      seenItems1
    } else {
      Set[String]()
    }

    // get the latest constraint unavailableItems $set event
    val unavailableItems: Set[String] = try {
      val constr = LEventStore.findByEntity(
        appName = ap.appName,
        entityType = "constraint",
        entityId = "unavailableItems",
        eventNames = Some(Seq("$set")),
        limit = Some(1),
        latest = true,
        timeout = Duration(200, "millis")
      )
      if (constr.hasNext) {
        constr.next.properties.get[Set[String]]("items")
      } else {
        Set[String]()
      }
    } catch {
      case e: scala.concurrent.TimeoutException =>
        logger.error(s"Timeout when read set unavailableItems event." +
          s" Empty list is used. ${e}")
        Set[String]()
      case e: Exception =>
        logger.error(s"Error when read set unavailableItems event: ${e}")
        throw e
    }

    // combine query's blackList,seenItems and unavailableItems
    // into final blackList.
    query.blackList.getOrElse(Set[String]()) ++ seenItems ++ unavailableItems
  }

  /** Get recent events of the user on items for recommending similar items */
  def getRecentItems(query: Query): Set[String] = {
    // get latest 10 user view item events

    var recentItems:Set[String] = Set(null)
    for (i <- 0 to query.users.size-1) {
      var recentEvents = try {
        LEventStore.findByEntity(
          appName = ap.appName,
          // entityType and entityId is specified for fast lookup
          entityType = "user",
          entityId = query.users(i),
          eventNames = Some(ap.similarEvents),
          targetEntityType = Some(Some("item")),
          limit = Some(10),
          latest = true,
          // set time limit to avoid super long DB access
          timeout = Duration(200, "millis")
        )
      } catch {
        case e: scala.concurrent.TimeoutException =>
          logger.error(s"Timeout when read recent events." +
            s" Empty list is used. ${e}")
          Iterator[Event]()
        case e: Exception =>
          logger.error(s"Error when read recent events: ${e}")
          throw e
      }

      val recentItems2: Set[String] = recentEvents.map { event =>
        try {
          event.targetEntityId.get
        } catch {
          case e => {
            logger.error("Can't get targetEntityId of event ${event}.")
            throw e
          }
        }
      }.toSet

      if(i==0){
        recentItems = recentItems2
      }else {
        recentItems.intersect(recentItems2).toSet
      }
    }

    recentItems
  }

  /** Prediction for user with known feature vector */
  def predictKnownUser(
    userFeature: Array[Double],
    productModels: Map[Int, ProductModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert to parallel collection
      .filter { case (i, pm) =>
        pm.features.isDefined &&
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
      }
      .map { case (i, pm) =>
        // NOTE: features must be defined, so can call .get
        val s = dotProduct(userFeature, pm.features.get)
        // may customize here to further adjust score
        (i, s)
      }
      .filter(_._2 > 0) // only keep items with score > 0
      .seq // convert back to sequential collection

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  /** Default prediction when know nothing about the user */
  def predictDefault(
    productModels: Map[Int, ProductModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert back to sequential collection
      .filter { case (i, pm) =>
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
      }
      .map { case (i, pm) =>
        // may customize here to further adjust score
        (i, pm.count.toDouble)
      }
      .seq

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  /** Return top similar items based on items user recently has action on */
  def predictSimilarRecommendation(
    recentFeatures: Vector[Array[Double]],
    productModels: Map[Int, ProductModel],
    query: Query,
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert to parallel collection
      .filter { case (i, pm) =>
        pm.features.isDefined &&
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
      }
      .map { case (i, pm) =>
        val s = recentFeatures.map{ rf =>
          // pm.features must be defined because of filter logic above
          cosine(rf, pm.features.get)
        }.reduce(_ + _)
        // may customize here to further adjust score
        (i, s)
      }
      .filter(_._2 > 0) // keep items with score > 0
      .seq // convert back to sequential collection

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  /** item과 비슷한 similar items들 return */
  def predictSimilar(
                      queryList: Set[Int],
                      queryFeatures: Vector[Array[Double]],
                      productModels: Map[Int, ProductModel],
                      query: Query,
                      whiteList: Option[Set[Int]],
                      blackList: Set[Int]
                    ): Array[(Int, Double)] = {
    val indexScores: Map[Int, Double] = productModels.par // convert to parallel collection
      .filter { case (i, pm) =>
      pm.features.isDefined &&
        (!queryList.contains(i)) && // only similar
        isCandidateItem(
          i = i,
          item = pm.item,
          categories = query.categories,
          whiteList = whiteList,
          blackList = blackList
        )
    }
      .map { case (i, pm) =>
        val s = queryFeatures.map{ rf =>
          // pm.features must be defined because of filter logic above
          cosine(rf, pm.features.get)
        }.reduce(_ + _)
        // may customize here to further adjust score
        (i, s)
      }
      .filter(_._2 > 0) // keep items with score > 0
      .seq // convert back to sequential collection

    val ord = Ordering.by[(Int, Double), Double](_._2).reverse
    val topScores = getTopN(indexScores, query.num)(ord).toArray

    topScores
  }

  private
  def getTopN[T](s: Iterable[T], n: Int)(implicit ord: Ordering[T]): Seq[T] = {

    val q = PriorityQueue()

    for (x <- s) {
      if (q.size < n)
        q.enqueue(x)
      else {
        // q is full
        if (ord.compare(x, q.head) < 0) {
          q.dequeue()
          q.enqueue(x)
        }
      }
    }

    q.dequeueAll.toSeq.reverse
  }

  private
  def dotProduct(v1: Array[Double], v2: Array[Double]): Double = {    val size = v1.size
    var i = 0
    var d: Double = 0
    while (i < size) {
      d += v1(i) * v2(i)
      i += 1
    }
    d
  }

  private
  def cosine(v1: Array[Double], v2: Array[Double]): Double = {
    val size = v1.size
    var i = 0
    var n1: Double = 0
    var n2: Double = 0
    var d: Double = 0
    while (i < size) {
      n1 += v1(i) * v1(i)
      n2 += v2(i) * v2(i)
      d += v1(i) * v2(i)
      i += 1
    }
    val n1n2 = (math.sqrt(n1) * math.sqrt(n2))
    if (n1n2 == 0) 0 else (d / n1n2)
  }

  private
  def isCandidateItem(
    i: Int,
    item: Item,
    categories: Option[Set[String]],
    whiteList: Option[Set[Int]],
    blackList: Set[Int]
  ): Boolean = {
    // can add other custom filtering here
    whiteList.map(_.contains(i)).getOrElse(true) &&
    !blackList.contains(i) &&
    // filter categories
    categories.map { cat =>
      item.categories.map { itemCat =>
        // keep this item if has ovelap categories with the query
        !(itemCat.toSet.intersect(cat).isEmpty)
      }.getOrElse(false) // discard this item if it has no categories
    }.getOrElse(true)

  }

}
