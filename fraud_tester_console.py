import pandas as pd
import numpy as np
import os
import joblib 
from tensorflow.keras.models import load_model
from collections import Counter 
from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION ---
BASE_DIR = r"D:\fraud_detectation" 
MODELS_DIR = os.path.join(BASE_DIR, 'models') 
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_for_gui.joblib')

# --- Model and Scaler Loading ---
try:
    print("⏳ Loading Models and Scaler...")
    scaler = joblib.load(SCALER_PATH)
    model_supervised = joblib.load(os.path.join(MODELS_DIR, 'model_supervised.joblib'))
    model_unsupervised = joblib.load(os.path.join(MODELS_DIR, 'model_unsupervised.joblib'))
    cnn_model = load_model(os.path.join(MODELS_DIR, 'cnn_model.keras'))
    lstm_model = load_model(os.path.join(MODELS_DIR, 'lstm_model.keras'))
    meta_classifier = joblib.load(os.path.join(MODELS_DIR, 'meta_classifier.joblib'))
    print("✅ All Models Loaded Successfully!")

except Exception as e:
    print(f"\n❌ ERROR: Failed to load models or scaler. Ensure all training scripts ran successfully.")
    print(f"Details: {e}")
    exit()

# --- Prediction Logic (Max Voting) ---
def get_prediction_votes(scaled_features_array):
    """Returns individual votes from LR, LOF, and Ensemble DL."""
    single_feature = scaled_features_array.reshape(1, -1)
    single_feature_3d = np.expand_dims(single_feature, axis=2)
    
    # 1. Supervised (LR)
    lr_pred = model_supervised.predict(single_feature)[0]
    lr_prob = model_supervised.predict_proba(single_feature)[0, 1]
    
    # 2. Unsupervised (LOF) 
    lof_pred_raw = model_unsupervised.predict(single_feature)[0]
    lof_pred = 1 if lof_pred_raw == -1 else 0 
    lof_score = model_unsupervised.decision_function(single_feature)[0]
    
    # 3. Ensemble DL
    cnn_prob = cnn_model.predict(single_feature_3d, verbose=0)[0, 0]
    lstm_prob = lstm_model.predict(single_feature_3d, verbose=0)[0, 0]
    ensemble_input = np.array([[cnn_prob, lstm_prob]])
    ensemble_pred = meta_classifier.predict(ensemble_input)[0]
    
    predictions = [lr_pred, lof_pred, ensemble_pred] 
    final_pred = Counter(predictions).most_common(1)[0][0]
    
    return lr_pred, lof_pred, ensemble_pred, final_pred, lr_prob, lof_score


# --- Input and Testing Loop ---
def run_console_test():
    """Takes user input from console and prints detailed results."""
    
    FEATURE_NAMES = ['Amount', 'Old Bal Org', 'New Bal Org', 'Old Bal Dest', 'New Bal Dest']
    
    # --- UPDATED TYPE MAPPING ---
    # Col Order: [CASH_OUT, DEBIT, PAYMENT, TRANSFER]
    TYPE_MAPPING = {
        '1': {'name': 'TRANSFER', 'vector': [0.0, 0.0, 0.0, 1.0], 'risk': 'High'},
        '2': {'name': 'CASH_OUT', 'vector': [1.0, 0.0, 0.0, 0.0], 'risk': 'High'},
        '3': {'name': 'PAYMENT', 'vector': [0.0, 0.0, 1.0, 0.0], 'risk': 'Low'},
        '4': {'name': 'CASH_IN', 'vector': [0.0, 0.0, 0.0, 0.0], 'risk': 'Low'} # CASH_IN is base (all zeros) because it was dropped in get_dummies
    }
    
    print("\n=======================================================")
    print("       ADVANCED FRAUD DETECTION CONSOLE TEST           ")
    print("=======================================================")

    while True:
        print("\n--- Enter New Transaction Details ---")
        raw_values_5 = []
        try:
            # 1. Numerical Inputs
            for name in FEATURE_NAMES:
                value = float(input(f"Enter {name}: "))
                raw_values_5.append(value)
            
            # 2. Type Selection
            print("\nSelect Transaction Type:")
            print("1: TRANSFER (High Risk)")
            print("2: CASH_OUT (High Risk)")
            print("3: PAYMENT (Low Risk)")
            print("4: CASH_IN (Low Risk)")
            
            type_choice = input("Enter choice (1/2/3/4): ")
            
            if type_choice in TYPE_MAPPING:
                type_info = TYPE_MAPPING[type_choice]
                type_name = type_info['name']
                type_vector = type_info['vector']
            else:
                print("Invalid choice. Defaulting to TRANSFER.")
                type_info = TYPE_MAPPING['1']
                type_name = type_info['name']
                type_vector = type_info['vector']


        except ValueError:
            print("\n❌ Invalid input. Please enter only numeric values.")
            continue
        except EOFError:
            break

        # --- Data Processing (Matching Training Logic) ---
        numerical_data_5 = np.array(raw_values_5).reshape(1, -1)
        scaled_numerical = scaler.transform(numerical_data_5)
        
        STEP_VALUE_UNSCALED = 0.0 
        
        # Total 10 Features: [STEP (1)] + [SCALED NUMERICAL (5)] + [TYPE VECTOR (4)]
        final_scaled_features = np.hstack(([STEP_VALUE_UNSCALED], scaled_numerical[0], type_vector)).reshape(1, -1)

        # --- Prediction and Output ---
        lr_pred, lof_pred, dl_pred, final_pred, lr_prob, lof_score = get_prediction_votes(final_scaled_features)
        
        final_result_text = "FRAUDULENT (BLOCK ACTION)" if final_pred == 1 else "LEGITIMATE (PASS)"

        print("\n-------------------------------------------------------")
        print(f"| TRANSACTION TYPE: {type_name:<34} |")
        print("-------------------------------------------------------")
        
        print(f"| Final VOTE (Max Voting): {final_result_text:<25} |")
        print("-------------------------------------------------------")
        
        print(f"| Model Vote | Prediction | Confidence/Score |")
        print(f"|------------|------------|------------------|")
        print(f"| LR (Supervised) | {'FRAUD' if lr_pred==1 else 'LEGITIMATE'} | Probability: {lr_prob:.4f} |")
        print(f"| LOF (Unsupervised) | {'FRAUD' if lof_pred==1 else 'LEGITIMATE'} | Anomaly Score: {lof_score:.4f} |")
        print(f"| Ensemble DL | {'FRAUD' if dl_pred==1 else 'LEGITIMATE'} | Vote: {dl_pred} |")
        print("-------------------------------------------------------")

        if input("\nTest another transaction? (y/n): ").lower() != 'y':
            break

# --- Main Execution ---
if __name__ == "__main__":
    run_console_test()