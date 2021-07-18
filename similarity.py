# DATA420-21S1 Assignment 2
# Audio Similarity

# setting up Spark
from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Question 1

# (a) produce descriptive statistics for
# each column in the chosen audio feature
spectral_temp = jmir_spectral_all.drop('MSD_TRACKID')

# view count, mean, stddev, min, max
spectral_temp.describe().toPandas().transpose()

# ------------------------------------------------------------------------------------------------

# create correlation matrix
# found help: https://stackoverflow.com/questions/52214404/how-to-get-the-correlation-matrix-of-a-pyspark-data-frame
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# create correlation vector column
assembler = VectorAssembler(inputCols = spectral_temp.columns, outputCol = "features")
spectral_vector = assembler.transform(spectral_temp).select("features")

# get correlation matrix
corr_matrix = Correlation.corr(spectral_vector, "features") 

# display readable table
import pandas as pd
corr_df = pd.DataFrame(corr_matrix.collect()[0][0].toArray().tolist())
corr_df

# ------------------------------------------------------------------------------------------------

# create dict of highly correlated features
high_corr = {}
for i in list(corr_df.columns):
    for j in list(corr_df.columns):
        if 1 > corr_df[i][j] > 0.7 and f"{j}, {i}" not in high_corr:
            high_corr[f"{i}, {j}"] = round(corr_df[i][j], 3)
            
high_corr

# ------------------------------------------------------------------------------------

# (b) load MSD All Music Genre Dataset (MAGD)
# visualise distribution of genres for matched songs

# load MAGD into dataframe
genre_schema = StructType([
    StructField("track_id", StringType(), True),
    StructField("genre", StringType(), True)])

genres = spark.read.csv("hdfs:///data/msd/genre/msd-MAGD-genreAssignment.tsv", sep = r'\t', schema = genre_schema)

# df with number of tracks per genre
genre_count = (
    genres.groupBy("genre")
    .count()
    .toPandas()
    .sort_values("count", ascending = True)
)

# ------------------------------------------------------------------------------------

# plot genre distribution for matched songs
import pandas as pd
import matplotlib.pyplot as plt

genre_count.plot(x = "genre", y = "count", kind = "barh", legend = False)
plt.xlabel("Number of Tracks")
plt.ylabel("Assigned Genre")
plt.savefig("genre_plot", bbox_inches = "tight", dpi = 400)

# ------------------------------------------------------------------------------------

# (c) Merge MAGD with audio features data

# remove extra quotes from track ID values
spectral = (
    jmir_spectral_all.withColumn("track_id",
        F.substring(F.col("MSD_TRACKID"), 2, 18))
    .drop("MSD_TRACKID")
)

# join datasets, only keep tracks with corresponding genre label
spectral = genres.join(jmir_spectral, "track_id", "inner")
spectral.count() # 420,620 rows

# ############################################################################################

# Question 2

# (b) create binary column to show
# whether genre is Electronic or not
# code adapted from CreditCardFraud.py

# add new binary column
genre_biclass = spectral.withColumn("class",
    F.when(F.col("genre") == "Electronic", 1).otherwise(0))

# find class balance
def print_class_balance(df):
    N = df.count()
    (
        df.groupBy("class")
        .count()
        .withColumn("ratio", F.round((F.col("count") / N), 3))
        .sort("class")
        .show(21)
    )

print_class_balance(genre_biclass)

# -----------------------------------------------------------------------------------------------

# (c) split data into training & test sets
# stratify to preserve class balance
# potentially use resampling methods
# code borrowed or adapted from CreditCardFraud.py]

# prepare data for stratified splitting
from pyspark.sql.window import Window

temp = (
    genre_biclass.withColumn("id", F.monotonically_increasing_id())
    .withColumn("Random", F.rand())
    .withColumn("Row", F.row_number()
        .over(Window.partitionBy("class").orderBy("Random"))
    )
)

# add vector containing features for classification
from pyspark.ml.feature import VectorAssembler

drop_cols = ["track_id", "genre", "class", "id", "Random", "Row"]
spectral_features = temp.drop(*drop_cols)
assembler = VectorAssembler(inputCols=spectral_features.columns, outputCol="features")
spectral_data = assembler.transform(temp)

# -----------------------------------------------------------------------------------------------

# extract training set
training = (
    spectral_data.where(
        ((F.col("class") == 0) & (F.col("Row") < 379954 * 0.8)) |
        ((F.col("class") == 1) & (F.col("Row") < 40666 * 0.8)))
    .drop(*spectral_features.columns)
)

training.cache()

# create test set
test = spectral_data.join(training, "id", "left_anti").drop(*spectral_features.columns)
test.cache()

# check class balance is preserved
print_class_balance(training)
print_class_balance(test)

# -----------------------------------------------------------------------------------------------

# negative class downsampling
# aiming for 1:4 ratio of positive class

pos_count = training.filter(F.col("class") == 1).count()
neg_count = training.filter(F.col("class") == 0).count()

training_downsampled = training.where(
        (F.col("class") != 0) |
        (F.col("class") == 0) & (F.col("Random") < 3 * (pos_count / neg_count))
)

print_class_balance(training_downsampled)

# -----------------------------------------------------------------------------------------------

# positive class upsampling
# aiming for 1:4 ratio of positive class
import numpy as np

ratio = 7.5
n = 10
p = ratio / n  # ratio < n such that probability < 1

def random_resample(x, n, p):
    return list(range((np.sum(np.random.random(n) > p)))) if x == 1 else [0]

random_resample_udf = F.udf(lambda x: random_resample(x, n, p), ArrayType(IntegerType()))

training_upsampled = (
    training
    .withColumn("sample", random_resample_udf(F.col("class")))
    .select(
        F.col("track_id"),
        F.col("features"),
        F.col("class"),
        F.explode(F.col("sample"))
        .alias("sample")
    )
    .drop("sample")
)

print_class_balance(training_upsampled)

# -----------------------------------------------------------------------------------------------

# observation re-weighting

training_weighted = training.withColumn(
    "Weight",
    F.when(F.col("class") == 1, 1.5)
    .otherwise(1.0)
)

print_class_balance(training_weighted)

# -------------------------------------------------------------------------------

# (d) train binary classification algorithms
# as chosen in Audio Similarity q2 (a)

# method 1: logistic regression
# code adapted from LogisticRegression.py
lr = LogisticRegression(maxIter=20, regParam=0.3, labelCol="class")
lr_model = lr.fit(training)

# fit using up & down-sampled data
lrd_model = lr.fit(training_downsampled)
lru_model = lr.fit(training_upsampled)

# fit using re-weighted observations
lrw = LogisticRegression(maxIter=20, regParam=0.3, labelCol="class", weightCol="Weight")
lrw_model = lrw.fit(training_weighted)

# -------------------------------------------------------------------------------

# method 2: random forest
rf = RandomForestClassifier(labelCol="class")
rf_model = rf.fit(training)

# fit using up & down-sampled data
rfu_model = rf.fit(training_upsampled)
rfd_model = rf.fit(training_downsampled)

# -------------------------------------------------------------------------------

# method 3: gradient-boosted trees
gbt = GBTClassifier(labelCol="class")
gbt_model = gbt.fit(training)

# fit using up & down-sampled data
gbtu_model = gbt.fit(training_upsampled)
gbtd_model = gbt.fit(training_downsampled)

# --------------------------------------------------------------------------------

# (e) Use test set to compute
# performance metrics for each model
# code borrowed from LogisticRegression.py

# setup for model testing

# auroc calculator
binary_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='rawPrediction',
    labelCol='class',
    metricName='areaUnderROC')

def get_metrics(pred):
    total = pred.count()

    nP = pred.filter((F.col('prediction') == 1)).count()
    nN = pred.filter((F.col('prediction') == 0)).count()
    TP = pred.filter((F.col('prediction') == 1) & (F.col('class') == 1)).count()
    FP = pred.filter((F.col('prediction') == 1) & (F.col('class') == 0)).count()
    FN = pred.filter((F.col('prediction') == 0) & (F.col('class') == 1)).count()
    TN = pred.filter((F.col('prediction') == 0) & (F.col('class') == 0)).count()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / total
    auroc = binary_evaluator.evaluate(pred)
    return [nP, nN, precision, recall, accuracy, auroc]


# --------------------------------------------------------------------------------

# test models and calculate metrics

models = {
    "logistic reg": lr_model,
    "lr upsampled": lru_model,
    "lr downsampled": lrd_model,
    "lr weighted": lrw_model,
    "random forest": rf_model,
    "rf upsampled": rfu_model,
    "rf downsampled": rfd_model,
    "gradient-boosted tree": gbt_model,
    "gbt upsampled": gbtu_model,
    "gbt downsampled": gbtd_model
}

for model in models:
    classifier = models[model]
    pred = classifier.transform(test)
    models[model] = get_metrics(pred)

# --------------------------------------------------------------------------------

# actual class count
nP_actual = pred.filter((F.col('class') == 1)).count()
nN_actual = pred.filter((F.col('class') == 0)).count()
print(f"Actual positive: {nP_actual}")
print(f"Actual negative: {nN_actual}")

# print metrics as pandas df
import pandas as pd
col_names = ["pos_pred", "neg_pred", "precision", "recall", "accuracy", "auroc"]
pd.DataFrame.from_dict(models, orient="index", columns=col_names)

# ############################################################################################

# Question 3

# (b) Cross-validation for tuning hyperparameters
# using gradient boosted tree classifier
# code adapted from https://towardsdatascience.com/multi-class-text-classification-with-pyspark-7d78d022ed35

# cross validation for number of trees and max depth
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.subsamplingRate, [0.25, 0.5, 0.75, 1])
             .addGrid(gbt.maxIter, [20, 50, 100])
             .addGrid(gbt.maxDepth, [3, 4, 5, 6])
             .build())
             
cv = CrossValidator(estimator=gbt, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=BinaryClassificationEvaluator(labelCol="class"), \
                    numFolds=3)

cv_model = cv.fit(training_upsampled)
cv_pred = cv_model.transform(test)

# ------------------------------------------------------------------------------

# print performance metrics

total = cv_pred.count()
nP = pred.filter((F.col('prediction') == 1)).count()
nN = pred.filter((F.col('prediction') == 0)).count()
TP = pred.filter((F.col('prediction') == 1) & (F.col('class') == 1)).count()
FP = pred.filter((F.col('prediction') == 1) & (F.col('class') == 0)).count()
FN = pred.filter((F.col('prediction') == 0) & (F.col('class') == 1)).count()
TN = pred.filter((F.col('prediction') == 0) & (F.col('class') == 0)).count()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / total
binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='class', metricName='areaUnderROC')
auroc = binary_evaluator.evaluate(pred)

print('total: {}'.format(total))
print('')
print('num positive actual: {}'.format(nP_actual))
print('num negative actual: {}'.format(nN_actual))
print('')
print('num positive: {}'.format(nP))
print('num negative: {}'.format(nN))
print('precision: {}'.format(TP / (TP + FP)))
print('recall: {}'.format(TP / (TP + FN)))
print('accuracy: {}'.format((TP + TN) / total))
print('auroc: {}'.format(auroc))

# ------------------------------------------------------------------------------

# (b) Convert genre column into an integer index
from pyspark.ml.feature import StringIndexer
strIdx = StringIndexer(inputCol = "genre", outputCol = "class")
strIdx_fit = strIdx.fit(spectral)
genre_multiclass = strIdx_fit.transform(spectral)

print_class_balance(genre_multiclass)

# ----------------------------------------------------------------------------------

# (c) split into test & training
# perform multiclass classification & evaluate metrics
# borrowed code from Q2(c)

# prepare data for stratified splitting
from pyspark.sql.window import Window

temp_multi = (
    genre_multiclass.withColumn("id", F.monotonically_increasing_id())
    .withColumn("Random", F.rand())
    .withColumn("Row", F.row_number()
        .over(Window.partitionBy("class").orderBy("Random"))
    )
)

# add vector containing features for classification
from pyspark.ml.feature import VectorAssembler

drop_cols = ["track_id", "genre", "class", "id", "Random", "Row"]
spectral_features = temp_multi.drop(*drop_cols)
assembler = VectorAssembler(inputCols=spectral_features.columns, outputCol="features")
spectral_multi = assembler.transform(temp_multi)

# -----------------------------------------------------------------------------------

# extract training set

training_multi = spectral_multi.where(
    ((F.col("class") == 0) & (F.col("Row") < 237649 * 0.8)) |
    ((F.col("class") == 1) & (F.col("Row") < 40666 * 0.8)) |
    ((F.col("class") == 2) & (F.col("Row") < 20899 * 0.8)) |
    ((F.col("class") == 3) & (F.col("Row") < 17775 * 0.8)) |
    ((F.col("class") == 4) & (F.col("Row") < 17504 * 0.8)) |
    ((F.col("class") == 5) & (F.col("Row") < 14314 * 0.8)) |
    ((F.col("class") == 6) & (F.col("Row") < 14194 * 0.8)) |
    ((F.col("class") == 7) & (F.col("Row") < 11691 * 0.8)) |
    ((F.col("class") == 8) & (F.col("Row") < 8780 * 0.8)) |
    ((F.col("class") == 9) & (F.col("Row") < 6931 * 0.8)) |
    ((F.col("class") == 10) & (F.col("Row") < 6801 * 0.8)) |
    ((F.col("class") == 11) & (F.col("Row") < 6182 * 0.8)) |
    ((F.col("class") == 12) & (F.col("Row") < 5789 * 0.8)) |
    ((F.col("class") == 13) & (F.col("Row") < 4000 * 0.8)) |
    ((F.col("class") == 14) & (F.col("Row") < 2067 * 0.8)) |
    ((F.col("class") == 15) & (F.col("Row") < 1613 * 0.8)) |
    ((F.col("class") == 16) & (F.col("Row") < 1535 * 0.8)) |
    ((F.col("class") == 17) & (F.col("Row") < 1012 * 0.8)) |
    ((F.col("class") == 18) & (F.col("Row") < 555 * 0.8)) |
    ((F.col("class") == 19) & (F.col("Row") < 463 * 0.8)) |
    ((F.col("class") == 20) & (F.col("Row") < 200 * 0.8))
    ).drop(*spectral_features.columns)

training_multi.cache()
print_class_balance(training_multi)

# create test set
test_multi = spectral_multi.join(training_multi, "id", "left_anti").drop(*spectral_features.columns)
test_multi.cache()
print_class_balance(test_multi)

# -----------------------------------------------------------------------------------

# fit rf multiclass model 
rf_multi = RandomForestClassifier(labelCol="class")
rfmulti_model = rf_multi.fit(training_multi)

# test model
rfmulti_pred = rfmulti_model.transform(test_multi)
rfmulti_pred.cache()

# -----------------------------------------------------------------------------------

# setup for model testing
import numpy as np

binary_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol='rawPrediction',
    labelCol='class',
    metricName='areaUnderROC')

def class_metrics(pred, num):
    nP_actual = pred.filter((F.col("class") == num)).count()
    nP = pred.filter((F.col("prediction") == num)).count()
    TP = pred.filter((F.col('prediction') == num) & (F.col('class') == num)).count()
    FP = pred.filter((F.col('prediction') == num) & (F.col('class') != num)).count()
    FN = pred.filter((F.col('prediction') != num) & (F.col('class') == num)).count()
    TN = pred.filter((F.col('prediction') != num) & (F.col('class') != num)).count()

    precision = TP / (TP + FP) if TP + FP > 0 else np.NaN
    recall = TP / (TP + FN) if TP + FN > 0 else np.NaN
    accuracy = (TP + TN) / total
    return [nP_actual, nP, TP, precision, recall, accuracy]

classes = {
    "Pop Rock": 0,
    "Electronic": 1,
    "Rap": 2,
    "Jazz": 3,
    "Latin": 4,
    "RnB": 5,
    "International": 6,
    "Country": 7,
    "Religious": 8,
    "Reggae": 9,
    "Blues": 10,
    "Vocal": 11,
    "Folk": 12,
    "New Age": 13,
    "Comedy (Spoken)": 14,
    "Stage": 15,
    "Easy Listening": 16,
    "Avant Garde": 17,
    "Classical": 18,
    "Children": 19,
    "Holiday": 20
}

# -----------------------------------------------------------------------------------

# model performance metrics

for genre in classes:
    metrics = class_metrics(rfmulti_pred, classes[genre])
    classes[genre] = metrics

print("Random Forest Multiclass")
auroc = binary_evaluator.evaluate(rfmulti_pred)
print(f"model auroc: {auroc}")

# display metrics as pandas df
import pandas as pd
col_names = ["num_actual", "num_pred", "correct_pred", "precision", "recall", "accuracy"]
pd.DataFrame.from_dict(classes, orient="index", columns=col_names)












