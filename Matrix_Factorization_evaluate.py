
import pyspark
from pyspark import SQLContext, SparkConf

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import Row

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col, rand, isnan

# Set Spark Config
conf = SparkConf().setAppName("RecSys-Challenge-Submission-Generation").setMaster("yarn")
conf = (conf.set("deploy-mode","cluster")
       .set("spark.driver.memory","100g")
       .set("spark.executor.memory","100g")
       .set("spark.driver.cores","1")
       .set("spark.num.executors","100")
       .set("spark.executor.cores","4")
       .set("spark.driver.maxResultSize", "100g"))
sc = pyspark.SparkContext(conf=conf)
sql = SQLContext(sc)

def load_file(path, train=False, id_mappings=()):
    '''
    Load a file from the hdfs file system.
    Parameters
    ----------
    path: str
        Path to the file on the HDFS
    Return
    ------
        Spark Dataframe
    '''
    df = (sql.read
        .format("csv")
        .option("header", "false")
        .option("sep", "\x01")
        .load(datafile,  inferSchema="true")
        .repartition(1000)
        .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"))

    tweet2id = df.select("tweet_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
    tweet2id = tweet2id.toDF().withColumnRenamed("_1", "tweet_id_str").withColumnRenamed("_2", "tweet")
    user2id = df.select("engaging_user_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
    user2id = user2id.toDF().withColumnRenamed("_1", "user_id_str").withColumnRenamed("_2", "user")

    if train:
        return tweet2id, user2id
    else:


target_cols = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]


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



max_user_id = user2id.groupBy().max("user").collect()[0][0]
max_tweet_id = tweet2id.groupBy().max("tweet").collect()[0][0]

def create_index(old, new):
    if old == "user":
        max_val = max_user_id
    elif old == "tweet":
        max_val = max_tweet_id
    return when(col(old).isNull(), col(new) + max_val).otherwise(col(old))


val_df = val_df.withColumn("user", create_index("user", "user_new"))
val_df = val_df.withColumn("tweet", create_index("tweet", "tweet_new"))


models = {}

maxIter=10
regParam=0.001
rank=15

for target_col in target_cols:
    target_col = target_col[:-10]
    print("Training Model for {}".format(target_col))
    models[target_col] = ALSModel.load("hdfs:///user/e1553958/RecSys/models/" + target_col + "_als_model")
    
    
    # Evaluate the model by computing the RMSE on the test data
    val_df = models[target_col].transform(val_df)
    val_df = val_df.withColumnRenamed("prediction", target_col )


def fallback_prediction(x):
    return when(isnan(x), rand()).otherwise(col(x))

for target_col in target_cols:
    target_col = target_col[:-10]
    val_df = val_df.withColumn(target_col, fallback_prediction(target_col))
    val_df.select("tweet_id", "engaging_user_id",target_col ).write.option("header", "false").csv("hdfs:///user/e1553958/RecSys/val_result/"+target_col)


if __name__ == "__main__":
    train_file = "hdfs:///user/pknees/RSC20/training.tsv"