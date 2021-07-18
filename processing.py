# DATA420-21S1 Assignment 2
# Data Processing

# setting up Spark
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# ###########################################################################################

# Question 1
# (c) count number of rows in each dataset

# -------------------------------------------------------------------------------------------

# data/msd/audio/features/...
# jmir datasets

jmir_area_of_moments = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv")
)

jmir_lpc = jmir_area_of_moments = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv")
)

jmir_methods_of_moments = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv")
)

jmir_mfcc = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv")
)

jmir_spectral_all = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv")
)

jmir_spectral_derivatives = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv")
)

# get row counts - 994,623 rows in each
jmir_area_of_moments.count()
jmir_lpc.count()
jmir_methods_of_moments.count()
jmir_mfcc.count()
jmir_spectral_all.count()
jmir_spectral_derivatives.count()

# -------------------------------------------------------------------------------------------

# data/msd/audio/features/...
# marsyas dataset - 995,000 rows
marsyas_timbral = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-marsyas-timbral-v1.0.csv")
)

marsyas_timbral.count() # 995,001

# -------------------------------------------------------------------------------------------

# data/msd/audio/features/...
# rhythmic patterns datasets - 994,188 rows
rp_mvd = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-mvd-v1.0.csv")
)

rp_rh = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-rh-v1.0.csv")
)

rp_rp = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-rp-v1.0.csv")
)

rp_ssd = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-ssd-v1.0.csv")
)

rp_trh = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-trh-v1.0.csv")
)

rp_tssd = (
    spark.read.format("com.databricks.spark.csv")
    .load("hdfs:///data/msd/audio/features/msd-tssd-v1.0.csv")
)

# get row counts - 994,188 rows
rp_mvd.count()
rp_rh.count()
rp_rp.count()
rp_ssd.count()
rp_trh.count()
rp_tssd.count()

# -------------------------------------------------------------------------------------------

# data/msd/audio/statistics/sample_properties.csv.gz
stats = (
    spark.read.format("com.databricks.spark.csv")
    .options(header = True)
    .load("hdfs:///data/msd/audio/statistics/sample_properties.csv.gz")
)
stats.count() # 992,865 rows

# -------------------------------------------------------------------------------------------

# data/msd/genre
topgenre = spark.read.csv("hdfs:///data/msd/genre/msd-topMAGD-genreAssignment.tsv", sep = r'\t')
style = spark.read.csv("hdfs:///data/msd/genre/msd-MASD-styleAssignment.tsv", sep = r'\t')
genre = spark.read.csv("hdfs:///data/msd/genre/msd-MAGD-genreAssignment.tsv", sep = r'\t')

topgenre.count() # 406,427 rows
style.count() # 422,714 rows
genre.count() # 273,936 rows

# -------------------------------------------------------------------------------------------

# data/msd/main/summary
analysis = (
    spark.read.format("com.databricks.spark.csv")
    .options(header = True)
    .load("hdfs:///data/msd/main/summary/analysis.csv.gz")
)

metadata = (
    spark.read.format("com.databricks.spark.csv")
    .options(header = True)
    .load("hdfs:///data/msd/main/summary/metadata.csv.gz")
)

# 1,000,000 rows
analysis.count()
metadata.count()

# -------------------------------------------------------------------------------------------

# data/msd/tasteprofile/...
taste = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv")
)

mismatches = spark.read.text("hdfs:///data/msd/tasteprofile/mismatches/sid_mismatches.txt")

taste.count() # 45,795,100 rows
mismatches.count() # 19,094 rows

# ###########################################################################################

# Question 2
# (a) filter taste profile to remove mismatched songs

# schema for tasteprofile/triplets.tsv
tasteSchema = StructType([
    StructField("user_id", StringType(), True),
    StructField("song_id", StringType(), True),
    StructField("play_count", IntegerType(), True)
])

# read taste profile to dataframe
taste_temp = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .schema(tasteSchema)
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv")
)
# taste_temp.count() # 48,373,586

# looking at tasteprofile/mismatches
mismatches_raw = spark.read.text("hdfs:///data/msd/tasteprofile/mismatches/sid_mismatches.txt")

# editing dataframe cols
# song id, track id, 1st song, 2nd song
mismatches = mismatches_raw.select(
    mismatches_raw.value.substr(9, 18).alias('song_id'),
    mismatches_raw.value.substr(28, 18).alias('track_id')
)

# mismatches.count() # 19,094
# mismatches.agg(F.countDistinct("song_id")).show() # 18,913

# remove mismatches from taste profile
taste = taste_temp.join(mismatches, 'song_id', 'left_anti')

# taste.count() # 45,795,100 so 2,578,486 observations have been removed
# taste.agg(F.countDistinct("song_id")).show() # 378,309 unique songs

# --------------------------------------------------------------------------------------------------

# (b) load audio feature attribute names & types
#  & use them to define schemas for the features

# create schema from the attribute files
def create_schema(atts_path):
    with open(atts_path, "r") as atts_raw:
        atts = [line.strip().split(",") for line in atts_raw.readlines()]
    for i in range(len(atts)):
        atts[i][1] = DoubleType() if atts[i][1] in ["NUMERIC", "real"] else StringType()
    new_schema = StructType([StructField(atts[i][0], atts[i][1], True) for i in range(len(atts))])
    return new_schema

# create df from feature using attribute schema
def audiofeature_csv2df(filename):
    path = "hdfs:///data/msd/audio/features/" + filename
    atts_path = "/scratch-network/courses/2021/DATA420-21S1/data/msd/audio/attributes/" \
        + filename[:-3] + "attributes.csv"
    new_schema = create_schema(atts_path)
    df = spark.read.format("com.databricks.spark.csv").schema(new_schema).load(path)
    return df

# --------------------------------------------------------------------------------------------------

# load all features in as dfs
# jmir
jmir_area_of_moments = audiofeature_csv2df("msd-jmir-area-of-moments-all-v1.0.csv")
jmir_lpc = audiofeature_csv2df("msd-jmir-lpc-all-v1.0.csv")
jmir_methods_of_moments = audiofeature_csv2df("msd-jmir-methods-of-moments-all-v1.0.csv")
jmir_mfcc = audiofeature_csv2df("msd-jmir-mfcc-all-v1.0.csv")
jmir_spectral_all = audiofeature_csv2df("msd-jmir-spectral-all-all-v1.0.csv")
jmir_spectral_derivatives = audiofeature_csv2df("msd-jmir-spectral-derivatives-all-all-v1.0.csv")

# marsyas
marsyas_timbral = audiofeature_csv2df("msd-marsyas-timbral-v1.0.csv")

# rhythm patterns
rp_mvd = audiofeature_csv2df("msd-mvd-v1.0.csv")
rp_rh = audiofeature_csv2df("msd-rh-v1.0.csv")
rp_rp = audiofeature_csv2df("msd-rp-v1.0.csv")
rp_ssd = audiofeature_csv2df("msd-ssd-v1.0.csv")
rp_trh = audiofeature_csv2df("msd-trh-v1.0.csv")
rp_tssd = audiofeature_csv2df("msd-tssd-v1.0.csv")

# ###########################################################################################

