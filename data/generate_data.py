# data/generate_data.py
import numpy as np
import pandas as pd

def generate_synthetic_data(n_clients: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    age = np.random.randint(25, 80, n_clients)
    income = np.random.normal(120000, 30000, n_clients).astype(int)
    AUM = np.abs(np.random.normal(600000, 200000, n_clients)).astype(int)
    # Use a negative binomial distribution for overdispersion in transactions
    num_transactions = np.random.negative_binomial(10, 0.5, n_clients)
    engagement_score = np.random.beta(2, 5, n_clients)

    # Introduce non-linear interactions and random noise
    risk_indicator = (
        (age > 55).astype(int) * 0.35 + 
        (income < 100000).astype(int) * 0.25 +
        (AUM < 500000).astype(int) * 0.20 +
        (engagement_score < 0.4).astype(int) * 0.20 +
        np.random.normal(0, 0.05, n_clients)
    )
    Risk_Label = (risk_indicator > 0.5).astype(int)

    data = pd.DataFrame({
        'age': age,
        'income': income,
        'AUM': AUM,
        'num_transactions': num_transactions,
        'engagement_score': engagement_score,
        'Risk_Label': Risk_Label
    })
    return data

if __name__ == '__main__':
    df = generate_synthetic_data()
    df.to_csv('data/client_data.csv', index=False)
    print("Synthetic dataset generated with shape:", df.shape)
