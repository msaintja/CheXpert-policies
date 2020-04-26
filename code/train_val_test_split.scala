import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.split

var train = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("project/CheXpert-v1.0-small/train.csv")

// val valid = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("project/CheXpert-v1.0-small/valid.csv")

// here, 0.0 corresponds to a I-Zero policy (we set implicit negatives to 0, just like explicit negatives).
// to implement I-0.33, replace 0.0 with 0.33 below
train = train.na.fill(0.0)

// here, the 0.66 corresponds to a U-0.66 policy (we set uncertain labels to 0.66 (close to 1).
// to implement a U-Zeros or U-Ones policy, replace these values to 0 or 1.
// you may also chose to implement this differently per-disease (each following its own policy).
train = train.withColumn("No Finding", when(col("No Finding") === -1.0, 0.66).otherwise(col("No Finding"))).withColumn("Enlarged Cardiomediastinum", when(col("Enlarged Cardiomediastinum") === -1.0, 0.66).otherwise(col("Enlarged Cardiomediastinum"))).withColumn("Cardiomegaly", when(col("Cardiomegaly") === -1.0, 0.66).otherwise(col("Cardiomegaly"))).withColumn("Lung Opacity", when(col("Lung Opacity") === -1.0, 0.66).otherwise(col("Lung Opacity"))).withColumn("Lung Lesion", when(col("Lung Lesion") === -1.0, 0.66).otherwise(col("Lung Lesion"))).withColumn("Edema", when(col("Edema") === -1.0, 0.66).otherwise(col("Edema"))).withColumn("Consolidation", when(col("Consolidation") === -1.0, 0.66).otherwise(col("Consolidation"))).withColumn("Pneumonia", when(col("Pneumonia") === -1.0, 0.66).otherwise(col("Pneumonia"))).withColumn("Atelectasis", when(col("Atelectasis") === -1.0, 0.66).otherwise(col("Atelectasis"))).withColumn("Pneumothorax", when(col("Pneumothorax") === -1.0, 0.66).otherwise(col("Pneumothorax"))).withColumn("Pleural Effusion", when(col("Pleural Effusion") === -1.0, 0.66).otherwise(col("Pleural Effusion"))).withColumn("Pleural Other", when(col("Pleural Other") === -1.0, 0.66).otherwise(col("Pleural Other"))).withColumn("Fracture", when(col("Fracture") === -1.0, 0.66).otherwise(col("Fracture"))).withColumn("Support Devices", when(col("Support Devices") === -1.0, 0.66).otherwise(col("Support Devices")))

train = train.withColumn("patientID", split($"Path", "\\/").getItem(2))

val patientIDs = train.select(train("patientID")).distinct

var Array(trainpatientIDs, valpatientIDs) = patientIDs.randomSplit(Array(0.75, 0.25), seed = 6250L)

val traintrain = train.join(trainpatientIDs, Seq("patientID"), "inner").drop("patientID")
val trainval = train.join(valpatientIDs, Seq("patientID"), "inner").drop("patientID")

traintrain.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("project/CheXpert-v1.0-small/traintrain.csv")
trainval.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("project/CheXpert-v1.0-small/trainval.csv")