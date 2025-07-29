import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_clean_data
from recommendation import train_als_model, get_recommendations
from evaluation import evaluate_model

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/yelp_sample.csv')
    df = load_and_clean_data(data_path)

    model, spark, spark_df = train_als_model(df)
    rmse = evaluate_model(model, spark_df)
    print(f"Model RMSE: {rmse}")

    recommendations = get_recommendations(model, spark, spark_df)
    recommendations.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
