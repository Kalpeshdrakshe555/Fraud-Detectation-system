Advanced Fraud Detection System Prototype
This project demonstrates a multi-layered, hybrid Deep Learning (DL) and Machine Learning (ML) architecture to detect online financial fraud. The system uses Max Voting across three specialized models to achieve highly reliable transaction classification.

💾 Project Structure (Local Folder)
Ensure your main project folder (D:\fraud_detectation) contains these files and folders:
Item,                           Type,                                     Purpose
create_balanced_sample.py,     Script,                          Creates the 50/50 balanced dataset.
data_preparation.py,           Script,                          "Cleans data, scales features, and saves training files/scaler."
hybrid_ml_train.py,            Script,                          Trains LR and LOF models (Hybrid ML Layer).
ensemble_dl_train.py,          Script,                          Trains CNN and LSTM models (Deep Learning Layer).
fraud_tester_console.py,       Script,                          Final testing module (console-based).
original_payments_log.csv,     DATA,                            "The large, raw dataset (must be present)."
models/,                       Folder,                           "Stores all trained models (.joblib, .keras)."

Setup and Execution Guide
Follow these steps in order to successfully set up, train, and test the entire fraud detection pipeline.

Step 0: Environment Setup (One Time)
Make sure you have all necessary Python libraries installed, especially TensorFlow (for Deep Learning) and scikit-learn (for ML).

pip install pandas numpy scikit-learn tensorflow joblib

Step 1: Data Preparation Pipeline (Run 1 & 2)
This ensures the data is balanced and ready for training.

Script,                                      Command,                              Action
1. Create Sample,             python create_balanced_sample.py,      Creates the 16.4K balanced dataset (online_fraud_sample_balanced.csv).
2. Prepare Data,              python data_preparation.py,            "Processes data, saves the scaled feature sets (.npy files), and saves the scaler_for_gui.joblib."

Step 2: Model Training Pipeline (Run 3 & 4)
This trains the three voting models of our architecture.

Script,                                           Command,                                        Action
3. Hybrid ML Train,                python hybrid_ml_train.py,                  Trains and saves the Logistic Regression (LR) and Local Outlier Factor (LOF) models. Prints F1-Scores and Blocked/Flagged summaries.
4. Ensemble DL Train,              python ensemble_dl_train.py,                Trains and saves the CNN and LSTM models. Prints detailed F1-Scores for each base model and the final Ensemble (Meta-Classifier).

Step 3: Final Testing and Verification (Run 5)
Use the console tester to verify that Max Voting works correctly for both low-risk and high-risk scenarios.

Script,                               Command,                                    Purpose
5. Run Tester,            python fraud_tester_console.py,         "Launches the interactive console test. You can input custom transactions and see the individual votes (LR, LOF, DL) and the Max Vote Final Decision."

Testing Scenarios (Must Check)
To prove your system works:
1.Low Risk Test (Type: PAYMENT / CASH_IN): Enter normal amounts. The output should be LEGITIMATE (PASS).
2.High Risk Test (Type: TRANSFER / CASH_OUT): Enter a high amount where Amount $\approx$ Old Bal Org and New Bal Org $\approx 0$. The output should be FRAUDULENT (BLOCK ACTION).

main dataset link :
https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset?resource=download

note : this project is under developement 
