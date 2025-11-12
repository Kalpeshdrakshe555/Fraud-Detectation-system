Project Title: Advanced Financial Fraud Detection Prototype
This project demonstrates a prototype of an advanced fraud detection system based on the provided research architecture, focusing on the Hybrid Machine Learning Approach (Supervised + Unsupervised Learning).

We use a sampled dataset derived from the Online Payments Fraud Detection Dataset to simulate real-world financial transactions.

Step 1: Data Acquisition & Preparation📌 
PurposeThe goal of this step is to clean, transform, and prepare the raw transaction data so it is suitable for training Machine Learning models. The final processed features are saved to a separate CSV file for easy identification and review.
⚙️ Working (Key Actions)
1.Feature Filtering: Irrelevant columns (nameOrig, nameDest, isFlaggedFraud) are removed.
2.Categorical Encoding: The type column (e.g., TRANSFER, CASH_OUT) is converted into a numerical format using One-Hot Encoding.
3.Scaling: All large numerical features (amount, balance columns) are scaled using StandardScaler to bring them to a uniform range (mean=0, std=1).
4.Output Save: The complete set of processed features is saved to the file processed_features_for_ml.csv.Data Split: The final processed data is divided into $80\%$ for Training (X_train) and $20\%$ for Testing (X_test).
code:
data_preparation.py