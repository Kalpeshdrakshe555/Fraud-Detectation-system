import pandas as pd
import os
import sys

# --- CONFIGURATION ---
ORIGINAL_BIG_FILE = "D:/fraud_detectation/dataset/PS_20174392719_1491204439457_log.csv" # Original, bade dataset ka naam
OUTPUT_FILE = "online_fraud_sample_balanced.csv"
BASE_DIR = r"D:\fraud_detectation" 

# --- Random State Handling ---
TEST_RANDOM_STATE = 42
if len(sys.argv) > 1:
    try:
        TEST_RANDOM_STATE = int(sys.argv[1])
    except ValueError:
        pass
print(f"--- Using RANDOM_STATE: {TEST_RANDOM_STATE} for sampling ---")

# --- Sample Creation Logic ---
try:
    print(f"⏳ Loading entire dataset from {ORIGINAL_BIG_FILE}...")
    df_full = pd.read_csv(os.path.join(BASE_DIR, ORIGINAL_BIG_FILE))

    # 1. Saare fraud samples nikaalna
    df_fraud = df_full[df_full['isFraud'] == 1].copy()
    num_fraud = len(df_fraud)
    
    # 2. Utne hi legitimate samples nikalna (50/50 ratio ke liye)
    df_legit = df_full[df_full['isFraud'] == 0].copy()
    
    # Random legitimate samples nikalna (Fraud ke barabar)
    df_legit_sample = df_legit.sample(n=num_fraud, random_state=TEST_RANDOM_STATE, replace=False)
    
    # 3. Final balanced dataset banana aur shuffle karna
    df_sample = pd.concat([df_fraud, df_legit_sample], ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=TEST_RANDOM_STATE).reset_index(drop=True) # Shuffle karna
    
    # 4. Save the new balanced file
    df_sample.to_csv(os.path.join(BASE_DIR, OUTPUT_FILE), index=False)
    
    print("-" * 50)
    print(f"✅ Success! BALANCED Dataset created: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_sample)} (Approx {num_fraud * 2}K)")
    print(f"Total Fraud Transactions: {df_sample['isFraud'].sum()} (50% Ratio)")
    print("-" * 50)

except FileNotFoundError:
    print(f"\n❌ Error: Original file '{ORIGINAL_BIG_FILE}' not found. Please verify the path.")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")