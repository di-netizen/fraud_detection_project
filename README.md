# 💳 Fraud Detection using Machine Learning

This project demonstrates how to identify fraudulent financial transactions using machine‑learning techniques.  
It includes synthetic data generation, preprocessing, visualization, model training, evaluation, and prediction using Random Forest.

---

## 📁 Project Structure


---

## 🔍 Project Overview

- **Goal**: Build a model to detect fraudulent transactions
- **Data**: Synthetic data with numerical and categorical features
- **Target**: `Is_Fraud` (0 = Legit, 1 = Fraud)
- **Model Used**: Random Forest Classifier

---

## 📊 Visualizations Included

- Class distribution
- Transaction amount by fraud status
- Fraud distribution by transaction type
- Feature correlation heatmap

---

## 📦 Python Libraries Used

Make sure the following libraries are installed:


> You can install them all with:
> ```bash
> pip install pandas numpy scikit-learn seaborn matplotlib joblib
> ```

---

## 🚀 How to Run the Project

```bash
# Step 1: Generate synthetic data
python scripts/generate_data.py

# Step 2: Clean and encode data
python scripts/clean_data.py

# Step 3: Visualize the data
python scripts/visualize_data.py

# Step 4: Split data into train/test
python scripts/split_data.py

# Step 5: Train and compare models
python scripts/train_models.py

# Step 6: Save model and simulate prediction
python scripts/final_model_predict.py




MIT License

Copyright (c) 2025 Divya Pawar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
