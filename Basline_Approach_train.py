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

from pyspark.ml.feature import StandardScaler, QuantileDiscretizer, StringIndexer, FeatureHasher, OneHotEncoderEstimator, CountVectorizer,PCA, VectorAssembler
from pyspark.ml import Pipeline

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col, rand, isnan, split, array

from pyspark.ml.classification import RandomForestClassifier

# Set Spark Config
conf = SparkConf().setAppName("RecSys-Challenge-Train-Model").setMaster("yarn")
conf = (conf.set("deploy-mode","cluster")
       .set("spark.driver.memory","200g")
       .set("spark.executor.memory","200g")
       .set("spark.driver.cores","1")
       .set("spark.num.executors","500")
       .set("spark.executor.cores","1")
       .set("spark.driver.maxResultSize", "200g"))
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
                "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"))

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

def create_quantilesDiscretizer(input_col: str, nq:int) -> QuantileDiscretizer:
    """
    Create a Quantile Discretizer for a specified column 
    Uses as output colum the input + _encoded
    
    Parameters
    ----------
    input_col: str
        Name of the Input Column
    nq: int
        Number of Quantiles to use
        
    Return
    ------
    QuantileDiscretizer
    """
    output_col = input_col + "_encoded"
    return QuantileDiscretizer(numBuckets=nq,
                                  relativeError=0.,
                                  handleInvalid='keep',
                                  inputCol=input_col,
                                  outputCol=output_col)

def create_stringIndexer(input_col:str) -> StringIndexer:
    """
    Create a String Indexer for a specified column 
    Uses as output colum the input + _encoded
    
    Parameters
    ----------
    input_col: str
        Name of the Input Column
        
    Return
    ------
    StringIndexer
    """
    output_col = input_col + "_encoded"
    return StringIndexer(inputCol=input_col,
                         outputCol=output_col,
                        handleInvalid='keep',)


def create_featureHasher(input_col:str, nq:int) -> FeatureHasher:
    """
    Create a Feature Hasher for a specified column 
    Uses as output colum the input + _encoded (creates oneHotEncodings for strings)
    
    Parameters
    ----------
    input_col: str
        Name of the Input Column
    nq: Int
        Number of Quantiles to use
        
        
    Return
    ------
    FeatureHasher
    """
    output_col = input_col + "_encoded"
    return FeatureHasher(numFeatures=nq,
                         inputCols=[input_col],
                         outputCol=output_col)


def create_countVectorizer(input_col: str) -> CountVectorizer:
    """
    Create a Count Vectorizer for a specified column 
    Uses as output colum the input + _encoded (Count Vectors for every column)
    
    Parameters
    ----------
    input_col: str
        Name of the Input Column
        
    Return
    ------
    FeatureHasher
    """
    output_col = input_col + "_encoded"
    return CountVectorizer(inputCol=input_col,
                           outputCol=output_col)

def encode_response(x):
    return when(col(x).isNull(), float(0)).otherwise(float(1))

if __name__ == "__main__":
    train_file = "hdfs:///user/pknees/RSC20/training.tsv"
    #train_file = "data/training_sample.tsv"
    train_df = load_file(train_file)

    numeric_cols = ['engaged_with_user_follower_count', 
                'engaged_with_user_following_count', 
                'engaged_with_user_account_creation',
                'engaging_user_follower_count', 
                'engaging_user_following_count',
                'engaging_user_account_creation',
                'tweet_timestamp',
               ]
    categorical_cols = ['tweet_type', 'language', 
                        'engaged_with_user_is_verified', 
                        'engaging_user_is_verified', 
                        'engaged_follows_engaging']
    id_cols = ['tweet_id', 'engaged_with_user_id', 'engaging_user_id']

    response_cols = ['reply_timestamp', 
                    'retweet_timestamp',
                    'retweet_with_comment_timestamp', 
                    'like_timestamp'
                    ]

    tweet_feature_cols = ['text_tokens', 'hashtags',
                        'present_media', 
                        'present_links', 
                        'present_domains']
    
    nq = 50 # number of quantiles to use

    # Encode Numeric Features (5.3.1)
    quantile_discretizers_numeric = [ create_quantilesDiscretizer(col, nq) for col in numeric_cols ]
    # Encode Categorical Features (5.3.2)
    string_indexer_categorical = [ create_stringIndexer(col) for col in categorical_cols]
    # Encode ID Features (5.3.3)
    id_feature_hashers = [ create_featureHasher(col, nq) for col in id_cols]
    # Encode Tweet Features (5.3.4 + 5.3.5)
    #tweet_countVectorizers = [ create_countVectorizer(col) for col in tweet_feature_cols]

    encoded_columns = [ col + "_encoded" for col in numeric_cols ]
    encoded_columns.extend([ col + "_encoded" for col in categorical_cols ])
    encoded_columns.extend([ col + "_encoded" for col in id_cols ])
    #encoded_columns.extend([ col + "_encoded" for col in tweet_feature_cols ])

    feature_assambler = VectorAssembler(inputCols=encoded_columns,
                                               outputCol="features")

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
    

    # Encode Respones Columns to 0, 1
    # Create Random Forest for every response col
    models = []
    for column in response_cols:
        train_df = train_df.withColumn(column, encode_response(column))

        models.append(RandomForestClassifier(labelCol=column, featuresCol="scaledFeatures", numTrees=15)
                      .setPredictionCol(column+"_pred")
                      .setRawPredictionCol(column+"_pred_raw")
                      .setProbabilityCol(column+"_proba"))

    
    # create a list of all transformers
    stages = list()
    stages.extend(quantile_discretizers_numeric)
    stages.extend(string_indexer_categorical)
    stages.extend(id_feature_hashers)
    #stages.extend(tweet_countVectorizers)
    stages.append(feature_assambler)
    stages.append(scaler)
    stages.extend(models)
    # Create Pipeline
    pipeline = Pipeline(stages=stages)

    # Fit Pipeline and transform df
    pipeline = pipeline.fit(train_df)

    #pipeline.save("pipeline")
    pipeline.save("hdfs:///user/e1553958/RecSys/pipeline")

    