
import pyspark
from pyspark import SQLContext, SparkConf

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col, rand, isnan


conf = SparkConf().setAppName("RecSys-Challenge-Submission-Generation").setMaster("yarn")
conf = (conf.set("deploy-mode","cluster")
       .set("spark.driver.memory","50g")
       .set("spark.executor.memory","50g")
       .set("spark.driver.cores","1")
       .set("spark.num.executors","50")
       .set("spark.executor.cores","1")
       .set("spark.driver.maxResultSize", "50g"))

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

train_df = train_df.select("user", "tweet", "like","reply", "retweet", "retweet_with_comment" )


datafile_val = "hdfs:///user/pknees/RSC20/val.tsv"

val_df = (sql.read
    .format("csv")
    .option("header", "false")
    .option("sep", "\x01")
    .load(datafile_val,  inferSchema="true")
    .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging"))


val_df = val_df.select("tweet_id","engaging_user_id")

tweet2id_val = val_df.select("tweet_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
user2id_val = val_df.select("engaging_user_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
tweet2id_val = tweet2id_val.toDF().withColumnRenamed("_1", "tweet_id_str_val").withColumnRenamed("_2", "tweet_new")
user2id_val = user2id_val.toDF().withColumnRenamed("_1", "user_id_str_val").withColumnRenamed("_2", "user_new")

val_df = val_df.join(tweet2id_val, col("tweet_id") == col("tweet_id_str_val"), "left_outer")
val_df = val_df.join(user2id_val, col("engaging_user_id") == col("user_id_str_val"), "left_outer")

val_df = val_df.join(tweet2id, col("tweet_id") == col("tweet_id_str"), "left_outer")
val_df = val_df.join(user2id, col("engaging_user_id") == col("user_id_str"), "left_outer")




datafile_test = "hdfs:///user/pknees/RSC20/test.tsv"
test_df = (sql.read
    .format("csv")
    .option("header", "false")
    .option("sep", "\x01")
    .load(datafile_test,  inferSchema="true")
    .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging"))

test_df = test_df.select("tweet_id","engaging_user_id")

tweet2id_test = test_df.select("tweet_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
user2id_test = test_df.select("engaging_user_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
tweet2id_test = tweet2id_test.toDF().withColumnRenamed("_1", "tweet_id_str_test").withColumnRenamed("_2", "tweet_new")
user2id_test = user2id_test.toDF().withColumnRenamed("_1", "user_id_str_test").withColumnRenamed("_2", "user_new")

test_df = test_df.join(tweet2id_test, col("tweet_id") == col("tweet_id_str_test"), "left_outer")
test_df = test_df.join(user2id_test, col("engaging_user_id") == col("user_id_str_test"), "left_outer")

test_df = test_df.join(tweet2id, col("tweet_id") == col("tweet_id_str"), "left_outer")
test_df = test_df.join(user2id, col("engaging_user_id") == col("user_id_str"), "left_outer")


max_user_id = user2id.groupBy().max("user").collect()[0][0]
max_tweet_id = tweet2id.groupBy().max("tweet").collect()[0][0]

def create_index(old, new):
    if old == "user":
        max_val = max_user_id
    elif old == "tweet":
        max_val = max_tweet_id
    return when(col(old).isNull(), col(new) + max_val).otherwise(col(old))


test_df = test_df.withColumn("user", create_index("user", "user_new"))
test_df = test_df.withColumn("tweet", create_index("tweet", "tweet_new"))


models = {}

maxIter=10
regParam=0.001
rank=15

for target_col in target_cols:
    target_col = target_col[:-10]
    print("Training Model for {}".format(target_col))
    models[target_col] = ALS(maxIter=maxIter, regParam=regParam, rank=rank, 
          userCol="user", itemCol="tweet", ratingCol=target_col,
          coldStartStrategy="nan", implicitPrefs=True).fit(train_df)
    
    # Evaluate the model by computing the RMSE on the test data
    test_df = models[target_col].transform(test_df)
    test_df = test_df.withColumnRenamed("prediction", target_col )
    val_df = models[target_col].transform(val_df)
    val_df = val_df.withColumnRenamed("prediction", target_col )
    


def fallback_prediction(x):
    return when(isnan(x), rand()).otherwise(col(x))

for target_col in target_cols:
    target_col = target_col[:-10]
    test_df = test_df.withColumn(target_col, fallback_prediction(target_col))
    test_df.select("tweet", "user",target_col ).write.option("header", "false").csv("hdfs:///user/e1553958/"+target_col+"_test")
    val_df = val_df.withColumn(target_col, fallback_prediction(target_col))
    val_df.select("tweet", "user",target_col ).write.option("header", "false").csv("hdfs:///user/e1553958/"+target_col+"_val")
