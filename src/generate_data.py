import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_growth_data(n_days=365, seed=42):
    np.random.seed(seed)
    
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    df = pd.DataFrame({
        "date": dates,
        "sessions": np.random.randint(1000, 5000, n_days),
        "users": np.random.randint(800, 4000, n_days),
        "new_users": np.random.randint(200, 1000, n_days),
        "pageviews": np.random.randint(3000, 15000, n_days),
        "bounce_rate": np.random.uniform(0.3, 0.7, n_days),
        "avg_session_duration": np.random.uniform(60, 300, n_days),
        "conversions": np.random.randint(10, 200, n_days),
        "revenue": np.random.uniform(500, 5000, n_days),
        "channel": np.random.choice(["organic", "paid", "social", "email", "direct"], n_days)
    })
    
    return df

if __name__ == "__main__":
    df = generate_growth_data()
    df.to_csv("data/raw/growth_data.csv", index=False)
    print(f"Dataset generado: {df.shape}")
    print(df.head())