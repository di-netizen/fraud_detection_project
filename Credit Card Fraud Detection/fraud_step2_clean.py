import pandas as pd

# Load dataset
df = pd.read_csv("fraud_data.csv")

# Check shape and missing values
print("📊 Shape:", df.shape)
print("\n🔍 Missing values:\n", df.isnull().sum())

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, drop_first=True)

# Show class distribution
print("\n⚖️ Class balance:\n", df_encoded['Is_Fraud'].value_counts())

# Save cleaned version
df_encoded.to_csv("fraud_data_cleaned.csv", index=False)
print("\n✅ Cleaned data saved as 'fraud_data_cleaned.csv'")
