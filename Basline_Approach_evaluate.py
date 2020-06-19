'''
This file loads the training data of the RecSys 2020 Challenge
and trains an ALS Collabrative Filtering Classifier.
'''
import pyspark
from pyspark import SQLContext, SparkConf

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

from pyspark.ml.feature import QuantileDiscretizer, StringIndexer, FeatureHasher, OneHotEncoderEstimator, CountVectorizer,PCA, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col, rand, isnan, split, array

from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

# Set Spark Config
conf = SparkConf().setAppName("RecSys-Challenge-Evaluate-Model").setMaster("yarn")
conf = (conf.set("deploy-mode","cluster")
       .set("spark.driver.memory","100g")
       .set("spark.executor.memory","100g")
       .set("spark.driver.cores","4")
       .set("spark.num.executors","100")
       .set("spark.executor.cores","4")
       .set("spark.driver.maxResultSize", "100g"))
sc = pyspark.SparkContext(conf=conf)
sql = SQLContext(sc)


def load_file(path):
    '''
    Load a file from the hdfs file system.
    Encode all Columns in the right Format:
        1. Convert boolean to int (0, 1)
        2. 
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
        .load(path,  inferSchema="true")
        .repartition(1000)
        .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging"))

    df = df.withColumn("engaged_with_user_is_verified",col("engaged_with_user_is_verified").cast("Integer"))
    df = df.withColumn("engaging_user_is_verified",col("engaging_user_is_verified").cast("Integer"))
    df = df.withColumn("engaged_follows_engaging",col("engaged_follows_engaging").cast("Integer"))

    # Split the string representations of lists
    ## Convert the text tokens to array of ints
    split_text = pyspark.sql.functions.split(df['text_tokens'], '\t')
    df = df.withColumn("text_tokens", split_text)

    ## Convert present media to array of strings
    split_text = pyspark.sql.functions.split(df['present_media'], '\t')
    df = df.withColumn("present_media", when(col('present_media').isNull(), array().cast("array<string>")).otherwise(split_text))

    ## Convert present links to array of strings
    split_text = pyspark.sql.functions.split(df['present_links'], '\t')
    df = df.withColumn("present_links", when(col('present_links').isNull(), array().cast("array<string>")).otherwise(split_text))

    ## Convert hashtags to array of strings
    split_text = pyspark.sql.functions.split(df['hashtags'], '\t')
    df = df.withColumn("hashtags", when(col('hashtags').isNull(), array().cast("array<string>")).otherwise(split_text))

    ## Convert present_domains to array of strings
    split_text = pyspark.sql.functions.split(df['present_domains'], '\t')
    df = df.withColumn("present_domains", when(col('present_domains').isNull(), array().cast("array<string>")).otherwise(split_text))
    
    return df

if __name__ == "__main__":
    val_file = "hdfs:///user/pknees/RSC20/val.tsv"
    #train_file = "data/training_sample.tsv"
    val_df = load_file(val_file)

    response_cols = ['reply_timestamp', 
                    'retweet_timestamp',
                    'retweet_with_comment_timestamp', 
                    'like_timestamp'
                    ]

    #pipeline = Pipeline.load("pipeline")
    pipeline = PipelineModel.load("hdfs:///user/e1553958/RecSys/pipeline")

    # Fit Pipeline and transform df
    val_df = pipeline.transform(val_df)


    for column in response_cols:
        # Write results to file
        val_df.select("tweet_id", "engaging_user_id",column ).write.option("header", "false").csv("hdfs:///user/e1553958/RecSys/val_result/"+target_col+"_rf")

    