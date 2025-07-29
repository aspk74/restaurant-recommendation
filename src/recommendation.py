from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def train_als_model(df):
    spark = SparkSession.builder.appName("RestaurantRecommendation").getOrCreate()
    spark_df = spark.createDataFrame(df)

    # ALS expects columns: user, item, rating
    als = ALS(
        maxIter=10,
        regParam=0.1,
        userCol="user_id",
        itemCol="business_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
    )

    model = als.fit(spark_df)
    return model, spark, spark_df

def get_recommendations(model, spark, spark_df, num_recs=5):
    users = spark_df.select("user_id").distinct()
    recs = model.recommendForUserSubset(users, num_recs)
    return recs
