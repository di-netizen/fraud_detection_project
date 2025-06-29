import pandas as pd
from sklearn.model_selection import train_test_split

# Load encoded data
df = pd.read_csv("fraud_data_cleaned.csv")

# Features and target
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Confirm shapes
print("✅ Train-test split complete")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
