import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


def split_train_val(csv_path, val_split=0.1):
    """
    Split the 90% train file into 80% train and 10% val
    
    Args:
        csv_path: Path to 90% train CSV
        val_split: Fraction for validation (default 0.1 = 10%)
    
    Returns:
        df_train: Training DataFrame (80%)
        df_val: Validation DataFrame (10%)
    """
    print("="*60)
    print("STEP 1: Splitting data into train (80%) and validation (10%)")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Calculate split index
    val_size = int(len(df) * val_split)
    train_size = len(df) - val_size
    
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()
    
    print(f"Train set: {df_train.shape} ({train_size/len(df)*100:.1f}%)")
    print(f"Val set:   {df_val.shape} ({val_size/len(df)*100:.1f}%)")
    
    # Save split datasets
    df_train.to_csv('results/data_train_80pct.csv', index=False)
    df_val.to_csv('results/data_val_10pct.csv', index=False)
    print("\nSaved: data_train_80pct.csv, data_val_10pct.csv")
    
    return df_train, df_val


def create_lstm_model(input_shape, output_size):
    """Create LSTM model"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_data(df, window_size=20):
    """Prepare data with sliding windows"""
    data = df.values
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    return np.array(X), np.array(y)


def train_lstm(df_train, window_size=20):
    """
    Train LSTM model on 80% training data
    
    Returns:
        model: Trained LSTM model
        X_train, y_train: Training sequences
    """
    print("\n" + "="*60)
    print("STEP 2: Training LSTM Model")
    print("="*60)
    
    # Prepare sequences
    X_train, y_train = prepare_data(df_train, window_size)
    print(f"Created {len(X_train)} training sequences")
    
    # Create and train model
    print("\nTraining LSTM...")
    model = create_lstm_model(
        input_shape=(window_size, df_train.shape[1]),
        output_size=df_train.shape[1]
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=1,
        validation_split=0.1  # Internal validation for monitoring
    )
    
    # Save model
    model.save('results/lstm_model.keras')
    print("\nLSTM model saved: lstm_model.keras")
    
    return model, X_train, y_train


def generate_l1_errors(model, df, window_size=20, dataset_name=""):
    """
    Generate predictions and L1 errors for a dataset
    
    Returns:
        predictions_df: Predictions DataFrame
        l1_errors_df: L1 errors DataFrame
        actuals_df: Actual values DataFrame
    """
    predictions = []
    actuals = []
    
    for i in range(len(df) - window_size):
        # Get window
        window = df.iloc[i:i+window_size].values
        window = window.reshape(1, window_size, df.shape[1])
        
        # Predict
        pred = model.predict(window, verbose=0)
        predictions.append(pred[0])
        
        # Get actual value
        actual = df.iloc[i+window_size].values
        actuals.append(actual)
    
    # Create DataFrames
    predictions_df = pd.DataFrame(predictions, columns=df.columns)
    actuals_df = pd.DataFrame(actuals, columns=df.columns)
    
    # Calculate L1 errors
    l1_errors_df = pd.DataFrame(
        np.abs(actuals_df.values - predictions_df.values),
        columns=df.columns
    )
    
    print(f"{dataset_name} - Predictions: {predictions_df.shape}, Mean L1: {l1_errors_df.values.mean():.6f}")
    
    return predictions_df, l1_errors_df, actuals_df


def evaluate_lstm(y_true, y_pred, dataset_name=""):
    """Calculate and print LSTM evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.6f}")
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}


def train_ocsvm(l1_train, scaler=None):
    """
    Train OCSVM on training L1 errors
    
    Returns:
        ocsvm: Trained OCSVM model
        scaler: Fitted scaler
    """
    print("\n" + "="*60)
    print("STEP 4: Training OCSVM")
    print("="*60)
    
    X_train = l1_train.values
    
    # Fit scaler
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    # Train OCSVM
    print("Training OCSVM on L1 errors...")
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    ocsvm.fit(X_train_scaled)
    
    # Save model and scaler
    joblib.dump(ocsvm, 'results/ocsvm_model.pkl')
    joblib.dump(scaler, 'results/ocsvm_scaler.pkl')
    print("OCSVM model and scaler saved")
    
    return ocsvm, scaler


def compute_threshold(ocsvm, scaler, l1_val, window_size=20):
    """
    Compute threshold using validation set (μ + 3σ)
    
    Returns:
        threshold: Computed threshold
        mu, sigma: Mean and std of malicious counts
        val_predictions: OCSVM predictions on validation set
    """
    print("\n" + "="*60)
    print("STEP 5: Computing Threshold on Validation Set")
    print("="*60)
    
    # Scale validation L1 errors
    X_val = l1_val.values
    X_val_scaled = scaler.transform(X_val)
    
    # Predict on validation set
    val_predictions = ocsvm.predict(X_val_scaled)
    val_scores = ocsvm.decision_function(X_val_scaled)
    
    print(f"Validation OCSVM predictions:")
    print(f"  Normal (1):     {np.sum(val_predictions == 1)}")
    print(f"  Anomalies (-1): {np.sum(val_predictions == -1)}")
    
    # Count malicious operations in each sliding window
    malicious_counts = []
    for i in range(len(val_predictions) - window_size + 1):
        window_preds = val_predictions[i:i+window_size]
        n_malicious = np.sum(window_preds == -1)
        malicious_counts.append(n_malicious)
    
    malicious_counts = np.array(malicious_counts)
    
    # Calculate μ and σ
    mu = np.mean(malicious_counts)
    sigma = np.std(malicious_counts)
    threshold = mu + 3 * sigma
    
    print(f"\nThreshold Calculation (3-sigma rule):")
    print(f"  Mean (mu):              {mu:.4f}")
    print(f"  Std Dev (sigma):        {sigma:.4f}")
    print(f"  Threshold (mu + 3*sigma): {threshold:.4f}")
    print(f"  Window size:            {window_size}")
    print(f"  Total windows:          {len(malicious_counts)}")
    print(f"  Min malicious/window:   {malicious_counts.min()}")
    print(f"  Max malicious/window:   {malicious_counts.max()}")
    
    return threshold, mu, sigma, val_predictions, val_scores, malicious_counts


def main(csv_path='data/plc-226-small_split-later_efw-4_train.csv', window_size=20):
    """
    Main pipeline: Split data, train LSTM and OCSVM, compute threshold
    """
    
    # Open log file with UTF-8 encoding
    log_file = open('results/pipeline_log.txt', 'w', encoding='utf-8')
    
    def log_print(msg):
        """Print and log simultaneously"""
        print(msg)
        log_file.write(msg + '\n')
    
    log_print("="*60)
    log_print("LSTM-OCSVM ANOMALY DETECTION PIPELINE")
    log_print("="*60)
    
    # Step 1: Split data
    df_train, df_val = split_train_val(csv_path, val_split=0.1)
    
    # Step 2: Train LSTM
    model, X_train, y_train = train_lstm(df_train, window_size)
    
    # Step 3: Generate L1 errors
    log_print("\n" + "="*60)
    log_print("STEP 3: Generating L1 Errors")
    log_print("="*60)
    
    # Train set L1 errors
    train_pred, l1_train, train_actual = generate_l1_errors(
        model, df_train, window_size, "Train Set"
    )
    l1_train.to_csv('results/l1_errors_train.csv', index=True)
    train_pred.to_csv('results/lstm_predictions_train.csv', index=True)
    
    # Validation set L1 errors
    val_pred, l1_val, val_actual = generate_l1_errors(
        model, df_val, window_size, "Val Set"
    )
    l1_val.to_csv('results/l1_errors_val.csv', index=True)
    val_pred.to_csv('results/lstm_predictions_val.csv', index=True)
    
    # Evaluate LSTM
    lstm_train_metrics = evaluate_lstm(train_actual.values, train_pred.values, "LSTM Train")
    lstm_val_metrics = evaluate_lstm(val_actual.values, val_pred.values, "LSTM Val")
    
    # Step 4: Train OCSVM
    ocsvm, scaler = train_ocsvm(l1_train)
    
    # Get OCSVM predictions on train set
    X_train_scaled = scaler.transform(l1_train.values)
    train_ocsvm_pred = ocsvm.predict(X_train_scaled)
    train_ocsvm_scores = ocsvm.decision_function(X_train_scaled)
    
    log_print(f"\nTrain OCSVM predictions:")
    log_print(f"  Normal (1):     {np.sum(train_ocsvm_pred == 1)}")
    log_print(f"  Anomalies (-1): {np.sum(train_ocsvm_pred == -1)}")
    
    # Save train OCSVM predictions
    train_ocsvm_df = pd.DataFrame({
        'prediction': train_ocsvm_pred,
        'decision_score': train_ocsvm_scores
    })
    train_ocsvm_df.to_csv('results/ocsvm_predictions_train.csv', index=True)
    
    # Step 5: Compute threshold
    threshold, mu, sigma, val_ocsvm_pred, val_ocsvm_scores, malicious_counts = compute_threshold(
        ocsvm, scaler, l1_val, window_size
    )
    
    # Save validation OCSVM predictions
    val_ocsvm_df = pd.DataFrame({
        'prediction': val_ocsvm_pred,
        'decision_score': val_ocsvm_scores
    })
    val_ocsvm_df.to_csv('results/ocsvm_predictions_val.csv', index=True)
    
    # Save threshold info
    threshold_df = pd.DataFrame({
        'metric': ['mean', 'std', 'threshold', 'window_size'],
        'value': [mu, sigma, threshold, window_size]
    })
    threshold_df.to_csv('results/threshold.csv', index=False)
    
    # Save window malicious counts
    window_counts_df = pd.DataFrame({'malicious_count': malicious_counts})
    window_counts_df.to_csv('results/window_malicious_counts_val.csv', index=True)
    
    # Create comprehensive summary
    log_print("\n" + "="*60)
    log_print("FINAL SUMMARY")
    log_print("="*60)
    
    log_print("\n--- Data Split ---")
    log_print(f"Train set: {df_train.shape}")
    log_print(f"Val set:   {df_val.shape}")
    
    log_print("\n--- LSTM Performance ---")
    log_print(f"Train - MSE: {lstm_train_metrics['mse']:.6f}, R²: {lstm_train_metrics['r2']:.6f}")
    log_print(f"Val   - MSE: {lstm_val_metrics['mse']:.6f}, R²: {lstm_val_metrics['r2']:.6f}")
    
    log_print("\n--- OCSVM Performance ---")
    log_print(f"Train - Anomalies: {np.sum(train_ocsvm_pred == -1)}/{len(train_ocsvm_pred)}")
    log_print(f"Val   - Anomalies: {np.sum(val_ocsvm_pred == -1)}/{len(val_ocsvm_pred)}")
    
    log_print("\n--- Detection Threshold ---")
    log_print(f"Mean (mu):              {mu:.4f}")
    log_print(f"Std Dev (sigma):        {sigma:.4f}")
    log_print(f"Threshold (mu + 3*sigma): {threshold:.4f}")
    log_print(f"Window size:            {window_size}")
    
    log_print("\n--- Files Saved ---")
    log_print("Data:")
    log_print("  - data_train_80pct.csv")
    log_print("  - data_val_10pct.csv")
    log_print("\nLSTM:")
    log_print("  - lstm_model.keras")
    log_print("  - lstm_predictions_train.csv")
    log_print("  - lstm_predictions_val.csv")
    log_print("  - l1_errors_train.csv")
    log_print("  - l1_errors_val.csv")
    log_print("\nOCSVM:")
    log_print("  - ocsvm_model.pkl")
    log_print("  - ocsvm_scaler.pkl")
    log_print("  - ocsvm_predictions_train.csv")
    log_print("  - ocsvm_predictions_val.csv")
    log_print("\nThreshold:")
    log_print("  - threshold.csv")
    log_print("  - window_malicious_counts_val.csv")
    log_print("  - pipeline_log.txt")
    
    log_print("\n" + "="*60)
    log_print("PIPELINE COMPLETE!")
    log_print("="*60)
    
    log_file.close()
    
    print("\nAll results saved to results/ directory")
    print("Check pipeline_log.txt for complete summary")


if __name__ == "__main__":
    main(
        csv_path='data/processed_final_226_snd_train__table_3_time_and_state_EFD_7_train.csv',
        window_size=20
    )

# if __name__ == "__main__":
#     main(
#         csv_path='data/plc_161_sourse_and_destination_train.csv',
#         window_size=20
#     )