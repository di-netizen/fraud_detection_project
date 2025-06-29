import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("fraud_data_cleaned.csv")
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# Split for training (again just to ensure consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, 'fraud_rf_model.pkl')
print("✅ Random Forest model saved as 'fraud_rf_model.pkl'")

# ------------------------------
# Simulate a new transaction
# ------------------------------

# Get expected columns from training
expected_cols = X.columns.tolist()

# Manually define the input (example)
input_data = {
    'Amount': 9250,
    'Transaction_Type_POS': 1,
    'Transaction_Type_Online': 0,
    'Country_India': 0,
    'Country_UK': 1,
    'Country_US': 0,
    'Time_Morning': 0,
    'Time_Afternoon': 0,
    'Time_Evening': 1,
    'Time_Night': 0
}

# Ensure all expected columns are included
for col in expected_cols:
    if col not in input_data:
        input_data[col] = 0

# Make sure correct order
new_transaction = pd.DataFrame([input_data])[expected_cols]

# Load model
model = joblib.load('fraud_rf_model.pkl')

# Predict
prediction = model.predict(new_transaction)[0]
print("🚨 Predicted Fraud Status:", "FRAUD" if prediction == 1 else "Legit")
