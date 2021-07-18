# DATA420-21S1 Assignment 2
# Song Recommendations

# setting up Spark
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics

import numpy as np

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# using taste df created in Q2(a)
# Question 1

# ------------------------------------------------------------------

# (a) number of unique songs & users
taste.agg(F.countDistinct("song_id")).show() # 378,309 songs
taste.agg(F.countDistinct("user_id")).show() # 1,019,318 users

# ------------------------------------------------------------------

# (b) number of songs played by most active user

active_user = taste.groupBy("user_id").sum("play_count").sort("sum(play_count)", ascending=False).head()[0]
taste.filter(F.col("user_id") == active_user).count() # 195

# as a % of total unique songs
(195 / 378309) * 100  # 0.05%

# ------------------------------------------------------------------

# (c) distribution of song popularity

import pandas as pd
import matplotlib.pyplot as plt

song_pop = taste.groupBy("song_id").sum("play_count").toPandas()
song_pop.plot(x="sum(play_count)", type="density")
plt.savefig("songpop_den", bbox_inches="tight", dpi=400)

# distribution of user activity

user_act = taste.groupBy("user_id").sum("play_count").toPandas()
user_act.plot(x="sum(play_count)", type="density")
plt.savefig("useract_den", bbox_inches="tight", dpi=400)

# ------------------------------------------------------------------

# (d) remove songs played less than N times

N = 100
M = 30

low_plays = (
    taste.groupBy("song_id")
    .sum("play_count")
    .filter(F.col("sum(play_count)") < N)
    .select(F.col("song_id"))
)

no_lowplays = taste.join(low_plays, "song_id", "left_anti")

# remove users with less than M different songs
low_songs = (
    taste.groupBy("user_id")
    .count()
    .filter(F.col("count") < M)
    .select(F.col("user_id"))
)

taste_clean = no_lowplays.join(low_songs, "user_id", "left_anti")
# taste_clean.count() # 33,108,653 user-song plays

# ------------------------------------------------------------------

# (e) shuffle data & split into training & test sets

temp = taste_clean.withColumn("random", F.rand()).orderBy("random")
training, test_temp = temp.randomSplit([0.7, 0.3], seed=100)

# remove users from test set who aren't in training set
train_users = training.select(F.col("user_id")).distinct()
test = test_temp.join(train_users, "user_id", "inner")

# 12 users were removed
test_temp.count() # 10,138,624
test.count() # 10,138,612

# ##################################################################

# using training & test from Q1 (e)
# code adapted from CollaborativeFiltering.py

# Question 2

# -----------------------------------------------------------

# (a) ALS for implicit matrix factorization model

from pyspark.ml.feature import StringIndexer

# create numeric user ID column
str_user = StringIndexer(inputCol="user_id", outputCol="user_int")
struser_fit = str_user.fit(training)
training1 = struser_fit.transform(training)

# create numeric song ID column
str_song = StringIndexer(inputCol="song_id", outputCol="song_int")
strsong_fit = str_song.fit(training1)
training2 = strsong_fit.transform(training1)

# fit ALS model
als = ALS(maxIter=5, regParam=0.01, userCol="user_int", itemCol="song_int", ratingCol="play_count", implicitPrefs=True)
als_model = als.fit(training2)

# test - create numeric user ID column
struser_fit = str_user.fit(test)
test1 = struser_fit.transform(test)

# test - create numeric song ID column
strsong_fit = str_song.fit(test1)
test2 = strsong_fit.transform(test1)
test2.cache()

# -----------------------------------------------------------

# (b) generate recommendations for some users

from random import randint

# select random users
maxuser = test2.agg(F.countDistinct("user_id")).head()[0]
rand_ints = [randint(0, maxuser) for i in range(10)]
user_pred = {}

# make recommendation for each random user
for rand_num in rand_ints:
    user = test2.filter(F.col("user_int") == rand_num)
    current_pred = als_model.transform(user).select(F.col("prediction"))
    user_pred[rand_num] = [row.prediction for row in current_pred.collect()]


# compare recommendations to actual songs played

def extract_songs(x):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x]

extract_songs_udf = F.udf(lambda x: extract_songs(x))

relevant_songs = (
   test2
    .select(F.col("user_int"), F.col("song_int"), F.col("play_count"))
    .groupBy('user_int')
    .agg(
        F.collect_list(
            F.array(
                F.col("song_int"),
                F.col("play_count")
            )
        ).alias('relevance')
    )
    .withColumn("relevant_songs", extract_songs_udf(F.col("relevance")))
    .select("user_int", "relevant_songs")
)

relevant_songs.cache()

# -----------------------------------------------------------

# (c) compute performance metrics

# extract recommended songs

def extract_songs_top_k(x, k):
    x = sorted(x, key=lambda x: -x[1])
    return [x[0] for x in x][0:k]

extract_songs_top_k_udf = F.udf(lambda x: extract_songs_top_k(x, k))

k = 5
topK = als_model.recommendForAllUsers(k)
topK.cache()

recommended_songs = (
    topK
    .withColumn("recommended_songs", extract_songs_top_k_udf(F.col("recommendations")))
    .select("user_int", "recommended_songs")
)

recommended_songs.cache()

combined = (
    recommended_songs.join(relevant_songs, 'user_int', 'inner')
    .rdd
    .map(lambda row: (row[1], row[2]))
)

# calculate ranking metrics
k = 10
rank_metrics = RankingMetrics(combined)

precAtK = rank_metrics.precisionAt(k)
ndcgAtK = rank_metrics.ndcgAt(k)
mapAtK = rank_metrics.meanAveragePrecisionAt(k)

print(f"Precision at {k}: {precAtK}")
print(f"NDCG at {k}: {ndcgAtK}")
print(f"Mean Average Precision at {k}: {mapAtK}")



