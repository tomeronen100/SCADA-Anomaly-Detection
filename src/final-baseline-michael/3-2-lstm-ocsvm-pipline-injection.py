import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


def inject_attacks(df, attack_col='inter_arrival_time', reduction_factor=0.5, 
                   attack_length=10, gap_length=290):
    """
    Inject attacks by reducing inter_arrival_time by 50%
    Pattern: 10 packets attacked, then 290 normal packets, repeat
    
    Args:
        df: Original DataFrame
        attack_col: Column to attack
        reduction_factor: Multiply column by this (0.5 = 50% reduction)
        attack_length: Number of consecutive packets to attack
        gap_length: Number of normal packets between attacks
    
    Returns:
        df_injected: DataFrame with attacks
        labels: Binary labels (0=normal, 1=attack)
    """
    print("="*60)
    print("STEP 1: Injecting Attacks")
    print("="*60)
    
    df_injected = df.copy()
    labels = np.zeros(len(df), dtype=int)
    
    # Inject attacks in pattern
    pattern_length = attack_length + gap_length
    n_attacks = 0
    
    for i in range(0, len(df), pattern_length):
        # Attack the next 10 packets
        attack_start = i
        attack_end = min(i + attack_length, len(df))
        
        # Reduce inter_arrival_time by 50%
        df_injected.loc[attack_start:attack_end-1, attack_col] *= reduction_factor
        
        # Label as attack
        labels[attack_start:attack_end] = 1
        
        n_attacks += (attack_end - attack_start)
    
    print(f"Total packets: {len(df)}")
    print(f"Attack packets: {n_attacks} ({n_attacks/len(df)*100:.1f}%)")
    print(f"Normal packets: {len(df)-n_attacks} ({(len(df)-n_attacks)/len(df)*100:.1f}%)")
    print(f"Attack pattern: {attack_length} attack, {gap_length} normal")
    print(f"Column attacked: '{attack_col}' (reduced by {(1-reduction_factor)*100:.0f}%)")
    
    return df_injected, labels


def prepare_data(df, window_size=20):
    """Prepare data with sliding windows"""
    data = df.values
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    return np.array(X), np.array(y)


def predict_with_frozen_models(df_test, labels, window_size=20, 
                               lstm_path='results/lstm_model.keras',
                               ocsvm_path='results/ocsvm_model.pkl',
                               scaler_path='results/ocsvm_scaler.pkl',
                               threshold_path='results/threshold.csv'):
    """
    Run test data through frozen LSTM and OCSVM models
    Apply sliding window detection with threshold
    
    Returns:
        predictions: Window-level predictions (0=normal, 1=attack)
        per_packet_predictions: Packet-level OCSVM predictions
        l1_errors: L1 errors from LSTM
    """
    print("\n" + "="*60)
    print("STEP 2: Running Through Frozen Models")
    print("="*60)
    
    # Load models
    print("Loading models...")
    lstm_model = keras.models.load_model(lstm_path)
    ocsvm_model = joblib.load(ocsvm_path)
    scaler = joblib.load(scaler_path)
    
    # Load threshold
    threshold_df = pd.read_csv(threshold_path)
    threshold = threshold_df[threshold_df['metric'] == 'threshold']['value'].values[0]
    print(f"Detection threshold: {threshold:.4f}")
    
    # Generate LSTM predictions and L1 errors
    print("\nGenerating LSTM predictions...")
    predictions_lstm = []
    actuals = []
    
    for i in range(len(df_test) - window_size):
        window = df_test.iloc[i:i+window_size].values
        window = window.reshape(1, window_size, df_test.shape[1])
        
        pred = lstm_model.predict(window, verbose=0)
        predictions_lstm.append(pred[0])
        
        actual = df_test.iloc[i+window_size].values
        actuals.append(actual)
    
    predictions_df = pd.DataFrame(predictions_lstm, columns=df_test.columns)
    actuals_df = pd.DataFrame(actuals, columns=df_test.columns)
    
    # Calculate L1 errors
    l1_errors = pd.DataFrame(
        np.abs(actuals_df.values - predictions_df.values),
        columns=df_test.columns
    )
    
    print(f"Generated {len(l1_errors)} L1 error vectors")
    print(f"Mean L1 error: {l1_errors.values.mean():.6f}")
    
    # Run OCSVM on L1 errors
    print("\nRunning OCSVM on L1 errors...")
    X_test = l1_errors.values
    X_test_scaled = scaler.transform(X_test)
    
    ocsvm_predictions = ocsvm_model.predict(X_test_scaled)
    ocsvm_scores = ocsvm_model.decision_function(X_test_scaled)
    
    n_anomalies = np.sum(ocsvm_predictions == -1)
    print(f"OCSVM predictions:")
    print(f"  Normal (1):     {np.sum(ocsvm_predictions == 1)}")
    print(f"  Anomalies (-1): {n_anomalies}")
    
    # Apply sliding window detection
    print(f"\nApplying sliding window detection (window size: {window_size})...")
    window_predictions = []
    window_labels = []
    
    # Adjust labels to match L1 errors indices (start from window_size)
    labels_adjusted = labels[window_size:]
    
    for i in range(len(ocsvm_predictions) - window_size + 1):
        # Get predictions in this window
        window_preds = ocsvm_predictions[i:i+window_size]
        n_malicious = np.sum(window_preds == -1)
        
        # If malicious count exceeds threshold, mark window as attack
        is_attack = 1 if n_malicious > threshold else 0
        window_predictions.append(is_attack)
        
        # Window label: 1 if ANY packet in window is attacked
        window_label = 1 if np.any(labels_adjusted[i:i+window_size] == 1) else 0
        window_labels.append(window_label)
    
    window_predictions = np.array(window_predictions)
    window_labels = np.array(window_labels)
    
    print(f"Total windows: {len(window_predictions)}")
    print(f"Windows predicted as attack: {np.sum(window_predictions == 1)}")
    print(f"Windows with actual attacks: {np.sum(window_labels == 1)}")
    
    return (window_predictions, window_labels, ocsvm_predictions, 
            ocsvm_scores, l1_errors, labels_adjusted)


def evaluate_detection(window_predictions, window_labels, output_file='results/evaluation_results.txt'):
    """
    Evaluate detection performance and save to file
    
    Args:
        window_predictions: Predicted labels for windows
        window_labels: True labels for windows
        output_file: Path to save results
    """
    print("\n" + "="*60)
    print("STEP 3: Evaluating Detection Performance")
    print("="*60)
    
    # Calculate metrics
    precision = precision_score(window_labels, window_predictions, zero_division=0)
    recall = recall_score(window_labels, window_predictions, zero_division=0)
    f1 = f1_score(window_labels, window_predictions, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(window_labels, window_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Create detailed report
    report = []
    report.append("="*60)
    report.append("ATTACK DETECTION EVALUATION RESULTS")
    report.append("="*60)
    
    report.append("\n--- Detection Metrics ---")
    report.append(f"Precision: {precision:.4f}")
    report.append(f"Recall:    {recall:.4f}")
    report.append(f"F1-Score:  {f1:.4f}")
    
    report.append("\n--- Confusion Matrix ---")
    report.append(f"True Negatives (TN):  {tn}")
    report.append(f"False Positives (FP): {fp}")
    report.append(f"False Negatives (FN): {fn}")
    report.append(f"True Positives (TP):  {tp}")
    
    report.append("\n--- Additional Metrics ---")
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    report.append(f"Accuracy:    {accuracy:.4f}")
    report.append(f"Specificity: {specificity:.4f}")
    
    report.append("\n--- Window Statistics ---")
    report.append(f"Total windows: {len(window_labels)}")
    report.append(f"Normal windows (actual): {np.sum(window_labels == 0)}")
    report.append(f"Attack windows (actual): {np.sum(window_labels == 1)}")
    report.append(f"Normal windows (predicted): {np.sum(window_predictions == 0)}")
    report.append(f"Attack windows (predicted): {np.sum(window_predictions == 1)}")
    
    report.append("\n" + "="*60)
    
    # Print to console
    for line in report:
        print(line)
    
    # Save to file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nResults saved to: {output_file}")
    
    return precision, recall, f1


def main():
    """
    Main pipeline for attack injection and detection evaluation
    """
    
    print("="*60)
    print("ATTACK INJECTION AND DETECTION PIPELINE")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")

    df_test = pd.read_csv('data/processed_final_226_snd_test__table_3_time_and_state_EFD_7_test.csv')

    print(f"Test data shape: {df_test.shape}")
    
    # Inject attacks
    df_injected, labels = inject_attacks(
        df_test,
        attack_col='inter_arrival_time',
        reduction_factor=0.5,
        attack_length=10,
        gap_length=290
    )
    
    # Save injected data and labels
    df_injected.to_csv('results/test_data_injected.csv', index=False)
    labels_df = pd.DataFrame({'label': labels})
    labels_df.to_csv('results/injection_labels.csv', index=True)
    print("\nSaved injected data and labels to results/")
    
    # Run through frozen models
    (window_predictions, window_labels, ocsvm_predictions, 
     ocsvm_scores, l1_errors, labels_adjusted) = predict_with_frozen_models(
        df_injected,
        labels,
        window_size=20,
        lstm_path='results/lstm_model.keras',
        ocsvm_path='results/ocsvm_model.pkl',
        scaler_path='results/ocsvm_scaler.pkl',
        threshold_path='results/threshold.csv'
    )
    
    # Save OCSVM predictions
    ocsvm_pred_df = pd.DataFrame({
        'ocsvm_prediction': ocsvm_predictions,
        'ocsvm_score': ocsvm_scores,
        'true_label': labels_adjusted
    })
    ocsvm_pred_df.to_csv('results/ocsvm_predictions.csv', index=True)
    
    # Save L1 errors
    l1_errors.to_csv('results/l1_errors_test.csv', index=True)
    
    # Save window predictions
    window_pred_df = pd.DataFrame({
        'window_prediction': window_predictions,
        'window_true_label': window_labels
    })
    window_pred_df.to_csv('results/window_predictions.csv', index=True)
    
    print("\nSaved predictions and L1 errors to results/")
    
    # Evaluate
    precision, recall, f1 = evaluate_detection(
        window_predictions,
        window_labels,
        output_file='results/evaluation_results.txt'
    )
    
    # Create summary
    print("\n" + "="*60)
    print("FILES SAVED IN results/")
    print("="*60)
    print("Data:")
    print("  - test_data_injected.csv")
    print("  - injection_labels.csv")
    print("\nPredictions:")
    print("  - ocsvm_predictions.csv")
    print("  - window_predictions.csv")
    print("  - l1_errors_test.csv")
    print("\nEvaluation:")
    print("  - evaluation_results.txt")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Final F1-Score: {f1:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()