'''
This file loads the training data of the RecSys 2020 Challenge
and trains an ALS Collabrative Filtering Classifier.
'''
import pyspark
from pyspark import SQLContext, SparkConf

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col, rand, isnan

# Set Spark Config
conf = SparkConf().setAppName("RecSys-Challenge-Train-Model").setMaster("yarn")
conf = (conf.set("deploy-mode","cluster")
       .set("spark.driver.memory","100g")
       .set("spark.executor.memory","100g")
       .set("spark.driver.cores","1")
       .set("spark.num.executors","100")
       .set("spark.executor.cores","4")
       .set("spark.driver.maxResultSize", "100g"))
sc = pyspark.SparkContext(conf=conf)
sql = SQLContext(sc)


def load_file(path):
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
    train_df = (sql.read
        .format("csv")
        .option("header", "false")
        .option("sep", "\x01")
        .load(path,  inferSchema="true")
        .repartition(1000)
        .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"))
    # select only relevant columns
    train_df = train_df.select("engaging_user_id", "tweet_id", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    
    # map id-string to unique numbers
    tweet2id = train_df.select("tweet_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
    tweet2id = tweet2id.toDF().withColumnRenamed("_1", "tweet_id_str").withColumnRenamed("_2", "tweet")
    user2id = train_df.select("engaging_user_id").rdd.map(lambda x: x[0]).distinct().zipWithUniqueId()
    user2id = user2id.toDF().withColumnRenamed("_1", "user_id_str").withColumnRenamed("_2", "user")

    # Join the data with the generated ids
    train_df = train_df.join(tweet2id, col("tweet_id") == col("tweet_id_str"))
    train_df = train_df.join(user2id, col("engaging_user_id") == col("user_id_str"))
    train_df = train_df.select("user", "tweet", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    
    return train_df, user2id, tweet2id

def encode_response(x):
    '''
    Encode a response columnm with 0 or 1
    Parameter
    ---------
    x: str
        Name of the column to encode
    Return
        Int: 0 if no response, 1 if response
    '''
    return when(col(x).isNull(), float(0)).otherwise(float(1))

def train_model(df, target_col, parameters, path):
    '''
    Train a model for a specific target column
    Parameter
    --------
    df: Spark DataFrame
        DataFrame to use for training
    target_col: str
        String Representation of the target_col to use
    parameters: dict
        Parameters to use for the ALS (=maxIter, regParam, rank)
    path: str
        Path on the HDFS to save the model
    Return
    ------
    None
        But Saves model to path
    '''
    maxIter=parameters["maxIter"]
    regParam=parameters["regParam"]
    rank=parameters["rank"]

    model = ALS(maxIter=maxIter, regParam=regParam, rank=rank, 
            userCol="user", itemCol="tweet", ratingCol=target_col,
            coldStartStrategy="nan", implicitPrefs=True).fit(df)

    model.save(path + target_col + "_als_model")
        

if __name__ == "__main__":
    target_cols = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

    train_file = "hdfs:///user/e1553958/RSC20/training.tsv"
    train_df, user2id, tweet2id = load_file(train_file)

    # save id_mappings to use for evaluation later
    user2id.write.save('hdfs:///user/e1553958/RecSys/mappings/user2id', format='parquet', mode='append')
    tweet2id.write.save('hdfs:///user/e1553958/RecSys/mappings/tweet2id', format='parquet', mode='append')

    # Encdoe the response to numeric attributes
    for target_col in target_cols:
        train_df = train_df.withColumn(target_col[:-10], encode_response(target_col))

    train_df = train_df.select("user", "tweet", "like","reply", "retweet", "retweet_with_comment" )

    parameters = {
        "maxIter": 20,
        "regParam": 0.001,
        "rank": 20
    }
    model_path = "hdfs:///user/e1553958/RecSys/datasplit/models/"
    for target_col in target_cols: # Train models
        target_col = target_col[:-10]
        train_model(train_df, target_col, parameters, model_path)