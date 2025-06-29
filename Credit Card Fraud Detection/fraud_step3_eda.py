import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load original data (with categories for better plots)
df = pd.read_csv("fraud_data.csv")

# Count of fraud vs normal
plt.figure(figsize=(6, 4))
sns.countplot(x='Is_Fraud', data=df, palette='Set2')
plt.title("Fraud vs Non-Fraud Transactions")
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'])
plt.tight_layout()
plt.show()

# Amount distribution by fraud
plt.figure(figsize=(8, 5))
sns.boxplot(x='Is_Fraud', y='Amount', data=df, palette='Set3')
plt.title("Transaction Amount by Fraud")
plt.tight_layout()
plt.show()

# Fraud count by Transaction Type
plt.figure(figsize=(7, 5))
sns.countplot(x='Transaction_Type', hue='Is_Fraud', data=df, palette='cool')
plt.title("Fraud by Transaction Type")
plt.tight_layout()
plt.show()
