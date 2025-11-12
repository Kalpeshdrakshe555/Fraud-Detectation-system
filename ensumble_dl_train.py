import pandas as pd
import numpy as np
import os
import joblib 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

# --- CONFIGURATION ---
BASE_DIR = r"D:\fraud_detectation" 
MODELS_DIR = os.path.join(BASE_DIR, 'models') 

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --- Load Data (FLOAT32 FIX) ---
try:
    X_train_np = np.load(os.path.join(BASE_DIR, 'X_train.npy'), allow_pickle=True)
    X_test_np = np.load(os.path.join(BASE_DIR, 'X_test.npy'), allow_pickle=True)
    y_train_raw = np.load(os.path.join(BASE_DIR, 'y_train.npy'), allow_pickle=True)
    y_test_raw = np.load(os.path.join(BASE_DIR, 'y_test.npy'), allow_pickle=True)
    
    X_train_np = np.asarray(X_train_np, dtype=np.float32)
    X_test_np = np.asarray(X_test_np, dtype=np.float32)
    y_train = np.asarray(y_train_raw, dtype=np.float32) 
    y_test = np.asarray(y_test_raw, dtype=np.float32)   
    
except FileNotFoundError:
    print("❌ ERROR: Data files not found. Please run 'data_preparation.py' first.")
    exit()

# --- 1. Data Reshaping for DL ---
X_train_3d = np.expand_dims(X_train_np, axis=2) 
X_test_3d = np.expand_dims(X_test_np, axis=2)
n_features = X_train_3d.shape[1] 

print(f"\n--- Step 5: Deep Learning Ensemble Start (10 features) ---")

# --- 2. CNN Model ---
def build_cnn_model():
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_features, 1)),
        Flatten(),
        Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()
print("⏳ Training CNN Model (Epochs=5)...")
cnn_model.fit(X_train_3d, y_train, epochs=5, batch_size=16, verbose=1) 
cnn_pred_prob = cnn_model.predict(X_test_3d).flatten()
cnn_model.save(os.path.join(MODELS_DIR, 'cnn_model.keras'))

print("\n--- Base CNN Model Evaluation ---")
cnn_pred_binary = (cnn_pred_prob > 0.5).astype(int)
print(classification_report(y_test, cnn_pred_binary))
print("-" * 50)


# --- 3. LSTM Model ---
def build_lstm_model():
    model = Sequential([
        LSTM(units=10, activation='relu', input_shape=(n_features, 1)), 
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_model = build_lstm_model()
print("⏳ Training LSTM Model (Epochs=5)...")
lstm_model.fit(X_train_3d, y_train, epochs=5, batch_size=16, verbose=1) 
lstm_pred_prob = lstm_model.predict(X_test_3d).flatten()
lstm_model.save(os.path.join(MODELS_DIR, 'lstm_model.keras'))

print("\n--- Base LSTM Model Evaluation ---")
lstm_pred_binary = (lstm_pred_prob > 0.5).astype(int)
print(classification_report(y_test, lstm_pred_binary))
print("-" * 50)


# --- 4. Ensemble (Stacking) ---
stacked_predictions_train = np.hstack((cnn_model.predict(X_train_3d).flatten().reshape(-1, 1), 
                                       lstm_model.predict(X_train_3d).flatten().reshape(-1, 1)))
stacked_predictions_test = np.hstack((cnn_pred_prob.reshape(-1, 1), 
                                      lstm_pred_prob.reshape(-1, 1)))

meta_classifier = LogisticRegression(solver='liblinear')
meta_classifier.fit(stacked_predictions_train, y_train) 
final_ensemble_pred = meta_classifier.predict(stacked_predictions_test)
joblib.dump(meta_classifier, os.path.join(MODELS_DIR, 'meta_classifier.joblib'))

print("✅ Deep Learning Ensemble Training & Saving Done!")
print("\n--- Final Ensemble Model Evaluation (Meta-Classifier) ---")
print(classification_report(y_test, final_ensemble_pred))
print("-" * 50)