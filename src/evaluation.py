from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_model(model, spark_df):
    predictions = model.transform(spark_df)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse
