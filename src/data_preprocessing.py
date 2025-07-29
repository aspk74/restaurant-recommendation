import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Basic cleaning example (filter nulls, etc.)
    df = df.dropna(subset=['user_id', 'business_id', 'rating'])
    # Convert rating to float if needed
    df['rating'] = df['rating'].astype(float)
    return df
