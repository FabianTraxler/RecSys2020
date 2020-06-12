import pyspark
from pyspark import SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import  when, col



sc = pyspark.SparkContext()
sql = SQLContext(sc)


datafile = "hdfs:///user/pknees/RSC20/training.tsv"

train_df = (sql.read
    .format("csv")
    .option("header", "false")
    .option("sep", "\x01")
    .load(datafile,  inferSchema="true")
    .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"))


datafile_val = "hdfs:///user/pknees/RSC20/val.tsv"

#test_df = (sql.read
#    .format("csv")
#    .option("header", "false")
#    .option("sep", "\x01")
#    .load(datafile_val,  inferSchema="true")
#    .toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains","tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count","engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
#               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified","engaging_user_account_creation", "engaged_follows_engaging"))

user_indexer = StringIndexer(inputCol="engaging_user_id", outputCol="user")
tweet_indexer = StringIndexer(inputCol="tweet_id", outputCol="tweet")

pipeline = Pipeline(stages=[user_indexer, tweet_indexer])
train_df = pipeline.fit(train_df).transform(train_df)

#test_df = pipeline.transform(test_df)


def encode_response(x):
    return when(col(x).isNull(), float(0)).otherwise(float(1))

def implicit_feedback(creation_time, interaction_time):
    return when(col(interaction_time).isNull(), float(0)).otherwise(col(interaction_time)-col(creation_time))

train_df = train_df.withColumn("like", encode_response("like_timestamp"))
#test_df = test_df.withColumn("like", encode_response("like_timestamp"))


train_df = train_df.select("user", "tweet", "like")

(training, val) = train_df.randomSplit([0.8, 0.2])

#test = test_df.select(("user", "tweet", "like"))

als = ALS(maxIter=10, regParam=0.01, rank=20, 
          userCol="user", itemCol="tweet", ratingCol="like",
          coldStartStrategy="drop", implicitPrefs=True)
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(val)

predictionAndLabels = predictions.rdd.map(lambda r: (r.prediction, r.like))


# Instantiate metrics object
metrics = BinaryClassificationMetrics(predictionAndLabels)

# Area under precision-recall curve
print("Area under PR = %s" % metrics.areaUnderPR)


#predictions = model.transform(val)

#predictions.coalesce(1).saveAsTextFile("hdfs:///user/e1553958/result.txt")

with open("AUC-PR.txt", "w") as file:
    file.write(metrics.areaUnderPR)