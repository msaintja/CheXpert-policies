# Chest X-Ray classification (CheXpert dataset)


# Setup
## Initial setup of Docker

Follow [the tutorial on the CSE 6250 website](http://www.sunlab.org/teaching/cse6250/spring2020/env/env-docker-compose.html#run-exec-and-ssh-how-to-access-the-environment) on how to pull and set up the standard CSE6250 Docker image.

Minor modification to the `docker-compose.yml`:  
Open filesharing ports, e.g. have `/mnt/shared/project` link to the folder on the host machine where the `CheXpert-v1.0-small` exists.

In the same folder where the `docker-compose.yml` is, run:

```
docker-compose up
```

Once everything is downloaded, you can spin your container with:
```
docker-compose start && docker-compose exec bootcamp bash
```

## Hadoop setup / HDFS initial load
```
# sudo su - hdfs
```
```
-bash-4.2$ hdfs dfs -mkdir -p /user/root
-bash-4.2$ hdfs dfs -chown root /user/root
-bash-4.2$ exit
```

Copy the dataset to HDFS:
```
# hdfs dfs -mkdir project
# hdfs dfs -put CheXpert-v1.0-small project
```

Please not that this initial step may take some time. If you want to monitor the copy of the ~10.7GB of data, you may use the following command on a distinct terminal:
```
# watch -n 0.5 -d "hdfs dfs -du -s -h project"
```

During the copy, you may encounter a warning such as:
```
20/04/04 20:04:00 WARN hdfs.DFSClient: Caught exception
java.lang.InterruptedException
        at java.lang.Object.wait(Native Method)
        at java.lang.Thread.join(Thread.java:1252)
        at java.lang.Thread.join(Thread.java:1326)
        at org.apache.hadoop.hdfs.DFSOutputStream$DataStreamer.closeResponder(DFSOutputStream.java:609)
        at org.apache.hadoop.hdfs.DFSOutputStream$DataStreamer.endBlock(DFSOutputStream.java:370)
        at org.apache.hadoop.hdfs.DFSOutputStream$DataStreamer.run(DFSOutputStream.java:546)
```

An optional patch is available on the [HDFS-10429 issue page](https://issues.apache.org/jira/browse/HDFS-10429).
## Upgrade spark version

Go to your the location of your spark installation:
```
# cd $SPARK_HOME/..
```

Download a newer version:
```
# wget http://www.gtlib.gatech.edu/pub/apache/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
```

Untar the archive:
```
# tar -xzf spark-2.4.5-bin-hadoop2.7.tgz
```

Declare your new installation:
```
# export SPARK_HOME=/usr/lib/spark-2.4.5-bin-hadoop2.7
```

Run `spark-shell`:
```
# $SPARK_HOME/bin/spark-shell --master "local[4]" --driver-memory 10G
```

Confirm your spark version to be >= 2.4
```
scala> sc.version >= "2.4"
```


## Installing BigDL

```
# cd /
# mkdir bigdl
# cd bigdl
# wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/dist-spark-2.4.0-scala-2.11.# 8-all/0.9.0/dist-spark-2.4.0-scala-2.11.8-all-0.9.0-dist.zip
# yum install zip -y
# yum install unzip -y
# unzip dist-spark-2.4.0-scala-2.11.8-all-0.9.0-dist.zip
# export BIGDL_HOME=/bigdl
# export BIGDL_JAR_NAME=`ls ${BIGDL_HOME}/lib/ | grep jar-with-dependencies.jar`
# export BIGDL_JAR="${BIGDL_HOME}/lib/$BIGDL_JAR_NAME"
# export BIGDL_CONF=${BIGDL_HOME}/conf/spark-bigdl.conf
# export SPARK_HOME=/usr/lib/spark-2.4.5-bin-hadoop2.7
```




# Preprocessing (Spark / HDFS)

## Image statistics

Run `image_statistics.scala` in `spark-shell` with the Databricks `spark-deep-learning` package to get the mean and std. dev per image channel:
```
# cd /mnt/shared/project/CheXpert-v1.0-small
# $SPARK_HOME/bin/spark-shell --master "local[4]" --driver-memory 10G --properties-file ${BIGDL_CONF} --jars ${BIGDL_JAR}  --conf spark.driver.extraClassPath=${BIGDL_JAR} --conf spark.executor.extraClassPath=${BIGDL_JAR} -i ../code/image_statistics.scala
```
The CSV will be saved under `/mnt/shared/project/CheXpert-v1.0-small/image_statistics.csv`.


## Train/Validation/Test split

Run `train_val_test_split.scala` in `spark-shell` to get a proper validation split:
```
# $SPARK_HOME/bin/spark-shell --master "local[4]" --driver-memory 10G -i ../code/train_val_test_split.scala
```

Copy over the resulting files from HDFS to your local filesystem storage:

```
# hdfs dfs -mv project/CheXpert-v1.0-small/traintrain.csv/*.csv project/CheXpert-v1.0-small/train_train.csv
# hdfs dfs -mv project/CheXpert-v1.0-small/trainval.csv/*.csv project/CheXpert-v1.0-small/train_valid.csv
# hdfs dfs -rm -r project/CheXpert-v1.0-small/traintrain.csv
# hdfs dfs -rm -r project/CheXpert-v1.0-small/trainval.csv
# cd /mnt/shared/project/CheXpert-v1.0-small/
# hdfs dfs -get project/CheXpert-v1.0-small/train_train.csv
# hdfs dfs -get project/CheXpert-v1.0-small/train_valid.csv
```

# Learning (PyTorch)

Run `CheXpert.py`.

You may need to install [Pytorch](https://pytorch.org/get-started/locally/), and install required modules, some of which are listed in the `requirements.txt` file.  

For example, using pip:
```
# pip install -r requirements.txt
```

Please note that we have currently found issues (also mentioned [here](https://stackoverflow.com/questions/60478862/how-to-avoid-runtimeerror-error-in-loadlibrarya-for-torch-cat)) with CUDA-enabled learning on Windows with the latest Pytorch versions (1.4.0) and Python 3.8.x. We recommend downgrading to Python 3.7.6 for now should you choose to use Windows.  

If you want to log the values (Loss/AUC/F1) from the training and validation process, you may do so with the following command:

```
# python -u CheXpert.py 2>&1 | tee CheXpert.log
```