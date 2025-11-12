import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os 
import sys
import joblib 

# --- CONFIGURATION ---
INPUT_FILE = "online_fraud_sample_balanced.csv" 
BASE_DIR = r"D:\fraud_detectation" 

# --- Random State Handling ---
TEST_RANDOM_STATE = 42
if len(sys.argv) > 1:
    try:
        TEST_RANDOM_STATE = int(sys.argv[1])
    except ValueError:
        pass
print(f"--- Using RANDOM_STATE: {TEST_RANDOM_STATE} ---")


# --- Data Loading ---
try:
    df = pd.read_csv(os.path.join(BASE_DIR, INPUT_FILE))
except FileNotFoundError:
    print(f"❌ Error: Input file '{INPUT_FILE}' not found. Please run 'create_balanced_sample.py' first.")
    exit()

print("--- Step 1: Data Preparation Start (BALANCED Data) ---")

# 1. Data Cleaning and Encoding
# 'nameOrig', 'nameDest', 'isFlaggedFraud' drop kiye
df_processed = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X = df_processed.drop('isFraud', axis=1) 
y = df_processed['isFraud']              
X = pd.get_dummies(X, columns=['type'], drop_first=True) # 4 categorical features banenge

# 2. Scaling (5 features)
numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# *** SCALER SAVE CODE ***
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_for_gui.joblib')
joblib.dump(scaler, SCALER_PATH)
print(f"✅ StandardScaler (5 features) saved to: {SCALER_PATH}")


# 3. Final Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=TEST_RANDOM_STATE, 
    stratify=y 
)

# *** IMPORTANT: Feature Order Check (Total 10 Features) ***
# 1 (step) + 5 (scaled numerical) + 4 (categorical) = 10 FEATURES
print(f"\n✅ Final Split Ready! Total Features: {len(X.columns)}")
print(f"Column Order: {X.columns.tolist()}")


# 4. Save the prepared data
np.save(os.path.join(BASE_DIR, 'X_train.npy'), X_train.values)
np.save(os.path.join(BASE_DIR, 'X_test.npy'), X_test.values)
np.save(os.path.join(BASE_DIR, 'y_train.npy'), y_train.values)
np.save(os.path.join(BASE_DIR, 'y_test.npy'), y_test.values)

print("-" * 40)
print(f"✅ ALL PREPARED DATA SAVED in: {BASE_DIR}")