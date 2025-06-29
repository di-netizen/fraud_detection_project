import pandas as pd
import numpy as np

np.random.seed(42)

# Generate synthetic transaction data
n = 1000
data = {
    'Amount': np.random.uniform(10, 10000, n),
    'Transaction_Type': np.random.choice(['Online', 'POS', 'ATM'], n),
    'Country': np.random.choice(['India', 'US', 'UK', 'Germany'], n),
    'Time': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n),
    'Is_Fraud': np.random.choice([0, 1], n, p=[0.95, 0.05])  # 5% fraud
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("fraud_data.csv", index=False)
print("✅ Generated 'fraud_data.csv' with shape:", df.shape)
print(df.head())

# Show class distribution
print("\nFraud class distribution:\n", df['Is_Fraud'].value_counts())
