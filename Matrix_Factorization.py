
import pyspark
from pyspark import SQLContext, SparkConf

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col


conf = SparkConf().setAppName("RecSys-Challenge-2020").setMaster("yarn")
conf = (conf.set("deploy-mode","cluster")
       .set("spark.driver.memory","100g")
       .set("spark.executor.memory","100g")
       .set("spark.driver.cores","1")
       .set("spark.num.executors","30")
       .set("spark.executor.cores","1")
       .set("spark.driver.maxResultSize", "100g"))

sc = pyspark.SparkContext(conf=conf)
sql = SQLContext(sc)


datafile = "hdfs:///user/pknees/RSC20/training.tsv"

train_df = (sql.read
    .format("csv")
    .option("header", "false")
    .option("sep", "\x01")
    .load(datafile,  inferSchema="true")
    .repartition(1000)
    .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"))

train_df = train_df.select("engaging_user_id", "tweet_id", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")


#datafile_val = "hdfs:///user/pknees/RSC20/val.tsv"

#test_df = (sql.read
#    .format("csv")
#    .option("header", "false")
#    .option("sep", "\x01")
#    .load(datafile_val,  inferSchema="true")
#    .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
#               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging"))

tweet2id = train_df.select("tweet_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
user2id = train_df.select("engaging_user_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()

tweet2id = tweet2id.toDF().withColumnRenamed("_1", "tweet_id_str").withColumnRenamed("_2", "tweet")
user2id = user2id.toDF().withColumnRenamed("_1", "user_id_str").withColumnRenamed("_2", "user")

train_df = train_df.join(tweet2id, col("tweet_id") == col("tweet_id_str"))
train_df = train_df.join(user2id, col("engaging_user_id") == col("user_id_str"))
train_df = train_df.select("user", "tweet", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")

target_cols = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

def encode_response(x):
    return when(col(x).isNull(), float(0)).otherwise(float(1))

for target_col in target_cols:
    train_df = train_df.withColumn(target_col[:-10], encode_response(target_col))

train_df = train_df.select("user", "tweet", "reply", "retweet", "retweet_with_comment", "like")

(training, test) = train_df.randomSplit([0.8, 0.2])
#test = test_df.select(("user", "tweet", "like"))
models = {}

maxIter=10
regParam=0.001
rank=20

for target_col in target_cols:
    target_col = target_col[:-10]
    print("Training Model for {}".format(target_col))
    models[target_col] = ALS(maxIter=maxIter, regParam=regParam, rank=rank, 
          userCol="user", itemCol="tweet", ratingCol=target_col,
          coldStartStrategy="drop", implicitPrefs=True).fit(training)
    
    # Evaluate the model by computing the RMSE on the test data
    test = models[target_col].transform(test)
    test = test.withColumnRenamed("prediction", target_col+"_pred", )


metrics = {}

for target_col in target_cols:
    target_col = target_col[:-10]
    predictionAndLabels = test.rdd.map(lambda r: (r[target_col+"_pred"], r[target_col]))
    metric = BinaryClassificationMetrics(predictionAndLabels)
    metrics[target_col] = metric.areaUnderPR
    print("For {}: Area under PR = {}".format(target_col, metrics[target_col]))

results = sc.parallelize(metrics.items())

results.coalesce(1).saveAsTextFile("hdfs:///user/e1553958/result_twitter")