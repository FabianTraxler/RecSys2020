
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

def load_file(path, mappings_path):
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
    val_df = (sql.read
        .format("csv")
        .option("header", "false")
        .option("sep", "\x01")
        .load(path,  inferSchema="true")
        .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging"))
    # Load id_string to id mappings from training
    user2id = sql.read.format('parquet').load(mappings_path+"user2id")
    tweet2id = sql.read.format('parquet').load(mappings_path+"tweet2id")

    # Select relevant columns
    val_df = val_df.select("tweet_id","engaging_user_id")
    # Create mapping from id_string to id
    tweet2id_val = val_df.select("tweet_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
    user2id_val = val_df.select("engaging_user_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
    tweet2id_val = tweet2id_val.toDF().withColumnRenamed("_1", "tweet_id_str_val").withColumnRenamed("_2", "tweet_new")
    user2id_val = user2id_val.toDF().withColumnRenamed("_1", "user_id_str_val").withColumnRenamed("_2", "user_new")
    # Join Mapping with Dataframe
    val_df = val_df.join(tweet2id_val, col("tweet_id") == col("tweet_id_str_val"), "left_outer")
    val_df = val_df.join(user2id_val, col("engaging_user_id") == col("user_id_str_val"), "left_outer")
    # Join Mapping from training data with Dataframe
    val_df = val_df.join(tweet2id, col("tweet_id") == col("tweet_id_str"), "left_outer")
    val_df = val_df.join(user2id, col("engaging_user_id") == col("user_id_str"), "left_outer")

    # Get the maximum IDs from training
    max_user_id = user2id.groupBy().max("user").collect()[0][0]
    max_tweet_id = tweet2id.groupBy().max("tweet").collect()[0][0]

    def create_index(old, new):
        """
        check if ID from training exits
        if yes then use this id
        if not use newly generated id
        """
        if old == "user":
            max_val = max_user_id
        elif old == "tweet":
            max_val = max_tweet_id
        return when(col(old).isNull(), col(new) + max_val).otherwise(col(old))


    val_df = val_df.withColumn("user", create_index("user", "user_new"))
    val_df = val_df.withColumn("tweet", create_index("tweet", "tweet_new"))

    return val_df


def fallback_prediction(x):
    """
    Make a random Guess if model made no predicitons
    """
    return when(isnan(x), rand()).otherwise(col(x))


if __name__ == "__main__":
    target_cols = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

    val_file = "hdfs:///user/pknees/RSC20/val.tsv"
    mapping_path = "hdfs:///user/e1553958/RecSys/mappings/"
    val_df = load_file(val_file, mapping_path)

    for target_col in target_cols:
        target_col = target_col[:-10]
        # Load model
        model = ALSModel.load("hdfs:///user/e1553958/RecSys/models/" + target_col + "_als_model")
        # Get Predictions of the model
        val_df = models[target_col].transform(val_df)
        val_df = val_df.withColumnRenamed("prediction", target_col )
        # Fallback prediction
        val_df = val_df.withColumn(target_col, fallback_prediction(target_col))
        # Write results to file
        val_df.select("tweet_id", "engaging_user_id",target_col ).write.option("header", "false").csv("hdfs:///user/e1553958/RecSys/val_result/"+target_col)