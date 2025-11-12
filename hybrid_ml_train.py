import pandas as pd
import numpy as np
import os
import joblib 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
BASE_DIR = r"D:\fraud_detectation" 
MODELS_DIR = os.path.join(BASE_DIR, 'models') 

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- Load Data ---
try:
    X_train_np = np.load(os.path.join(BASE_DIR, 'X_train.npy'), allow_pickle=True)
    X_test_np = np.load(os.path.join(BASE_DIR, 'X_test.npy'), allow_pickle=True)
    y_train_raw = np.load(os.path.join(BASE_DIR, 'y_train.npy'), allow_pickle=True)
    y_test_raw = np.load(os.path.join(BASE_DIR, 'y_test.npy'), allow_pickle=True)
    
    X_train = pd.DataFrame(X_train_np)
    X_test = pd.DataFrame(X_test_np)
    y_train = pd.Series(y_train_raw)
    y_test = pd.Series(y_test_raw)
    
except FileNotFoundError:
    print("❌ ERROR: Data files not found. Please run 'data_preparation.py' first.")
    exit()

# --- Step 2: Supervised Learning (Logistic Regression) ---
model_supervised = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000) 
model_supervised.fit(X_train, y_train)
y_pred_supervised = model_supervised.predict(X_test)
y_prob_supervised = model_supervised.predict_proba(X_test)[:, 1]
joblib.dump(model_supervised, os.path.join(MODELS_DIR, 'model_supervised.joblib'))

print("\n--- Step 2: Supervised Model (LR) Evaluation ---")
print(classification_report(y_test, y_pred_supervised))
print("-" * 50)


# --- Step 3: Unsupervised Learning (LOF) ---
model_unsupervised = LocalOutlierFactor(n_neighbors=20, novelty=True)
model_unsupervised.fit(X_train) 
anomaly_predictions = model_unsupervised.predict(X_test)
anomaly_scores = model_unsupervised.decision_function(X_test)
y_pred_unsupervised = [1 if pred == -1 else 0 for pred in anomaly_predictions]
joblib.dump(model_unsupervised, os.path.join(MODELS_DIR, 'model_unsupervised.joblib'))

print("\n--- Step 3: Unsupervised Model (LOF) Evaluation ---")
print(classification_report(y_test, y_pred_unsupervised))
print("-" * 50)


# --- Step 4: Decision Making & Trigger Actions ---
results_df = pd.DataFrame({
    'Actual_Fraud': y_test.reset_index(drop=True),
    'Supervised_Prob_Fraud': y_prob_supervised,
    'Unsupervised_Anomaly': y_pred_unsupervised,
    'Anomaly_Score': anomaly_scores
})

HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.5

def final_decision(row):
    if row['Supervised_Prob_Fraud'] >= HIGH_CONFIDENCE_THRESHOLD:
        return 'Fraudulent Transaction (Block Action)'
    elif row['Supervised_Prob_Fraud'] >= LOW_CONFIDENCE_THRESHOLD and row['Unsupervised_Anomaly'] == 1:
        return 'Suspicious Transaction (Generate Alert)'
    else:
        return 'Legitimate Transaction'

results_df['Final_Decision'] = results_df.apply(final_decision, axis=1)

# --- TESTING SUMMARY (FLAGGED/BLOCKED) ---
fraud_count = (results_df['Final_Decision'] == 'Fraudulent Transaction (Block Action)').sum()
suspicious_count = (results_df['Final_Decision'] == 'Suspicious Transaction (Generate Alert)').sum()
legit_count = (results_df['Final_Decision'] == 'Legitimate Transaction').sum()

print("\n--- Step 4: Hybrid ML Testing Summary ---")
print(f"Total Test Transactions: {len(results_df)}")
print(f"Transactions BLOCKED (Fraudulent): {fraud_count}")
print(f"Transactions FLAGGED (Suspicious): {suspicious_count}")
print(f"Transactions PASSED (Legitimate): {legit_count}")
print("-" * 50)