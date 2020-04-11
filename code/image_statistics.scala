import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._

import org.opencv.core._
import org.opencv.core.Core._

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

import java.io._

Engine.init
val sqlContext = new SQLContext(sc)

val training_loader = ImageFrame.read("project/CheXpert-v1.0-small/train/*/*", sc, 4)

val rdd = training_loader.asInstanceOf[DistributedImageFrame].rdd.take(2000)

val l_mu = ListBuffer[List[Double]]()
val l_sigma = ListBuffer[List[Double]]()

rdd foreach {x => var mu = new MatOfDouble(); var sigma = new MatOfDouble(); meanStdDev(x.opencvMat, mu, sigma); if (x.getChannel == 1) {l_mu += List.concat(mu.get(0,0).toList, mu.get(0,0).toList, mu.get(0,0).toList); l_sigma += List.concat(sigma.get(0,0).toList, sigma.get(0,0).toList, sigma.get(0,0).toList)} else {l_mu += List.concat(mu.get(0,0).toList, mu.get(1,0).toList, mu.get(2,0).toList); l_sigma += List.concat(sigma.get(0,0).toList, sigma.get(1,0).toList, sigma.get(2,0).toList)} }

val list_mus = sc.parallelize(l_mu.toList).toDF.map(x => (x.getAs[mutable.WrappedArray[Double]](0)(0), x.getAs[mutable.WrappedArray[Double]](0)(1), x.getAs[mutable.WrappedArray[Double]](0)(2))).rdd.reduce{ case ((a,b,c),(d,e,f)) => (a+d, b+e, c+f) }

val res_mu = (list_mus._3 / l_mu.length / 255, list_mus._2 / l_mu.length / 255, list_mus._1 / l_mu.length / 255)

val list_sigmas = sc.parallelize(l_sigma.toList).toDF.map(x => (x.getAs[mutable.WrappedArray[Double]](0)(0), x.getAs[mutable.WrappedArray[Double]](0)(1), x.getAs[mutable.WrappedArray[Double]](0)(2))).rdd.reduce{ case ((a,b,c),(d,e,f)) => (a+d, b+e, c+f) }

val res_sigma = (list_sigmas._3 / l_sigma.length / l_sigma.length / 255, list_sigmas._2 / l_sigma.length / l_sigma.length / 255, list_sigmas._1 / l_sigma.length / l_sigma.length / 255)

val file = "image_statistics.csv"
val writer = new BufferedWriter(new FileWriter(file))
List(res_mu._1.toString, ", ", res_mu._2.toString, ", ", res_mu._3.toString, ", ", res_sigma._1.toString, ", ", res_sigma._2.toString, ", ", res_sigma._3.toString).foreach(writer.write)
writer.close()
