# Install required libraries
!pip install pyedflib tensorflow scipy numpy matplotlib PyWavelets

# Import libraries
import numpy as np
import pyedflib  # For handling .edf files
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pywt  # For Discrete Wavelet Transform
import os

# Download the EEG dataset
!wget -r -N -c -np https://physionet.org/files/eegmat/1.0.0/

# Load and preprocess the data
file_path = '/content/physionet.org/files/eegmat/1.0.0/'  # Directory containing the .edf files
edf_files = [f for f in os.listdir(file_path) if f.endswith('.edf')]

# Define the fixed length (300 samples per channel)
fixed_length = 300
all_eeg_signals = []

# Read each .edf file and extract the first 19 EEG channels
for edf_file in edf_files:
    edf = pyedflib.EdfReader(file_path + edf_file)
    signals = []
    for i in range(19):
        signal = edf.readSignal(i)
        # Truncate or pad each signal to the fixed length
        if len(signal) > fixed_length:
            signal = signal[:fixed_length]  # Truncate
        elif len(signal) < fixed_length:
            signal = np.pad(signal, (0, fixed_length - len(signal)), 'constant')  # Zero-pad
        signals.append(signal)
    all_eeg_signals.append(np.array(signals))
    edf._close()

# Convert the list of signals to a numpy array
all_eeg_signals = np.array(all_eeg_signals)
print("Shape of all_eeg_signals:", all_eeg_signals.shape)  # (num_files, 19, 300)

# Apply Discrete Wavelet Transform (DWT) for Noise Reduction and Feature Extraction
def apply_dwt(signal):
    coeffs = pywt.wavedec(signal, 'db8', level=4)  # DWT with Daubechies (db8), level 4
    return coeffs  # Returns a list of coefficients for each band

# Apply DWT to each EEG channel in all files
dwt_features = []
for i in range(all_eeg_signals.shape[0]):
    channels = []
    for j in range(19):  # Loop over each channel
        channels.append(apply_dwt(all_eeg_signals[i, j]))
    dwt_features.append(channels)

# Prepare Data for CNN Input (Reshape and Format)
def prepare_input(dwt_features):
    processed_signals = []
    for subject_features in dwt_features:
        subject_data = []
        for channel_coeffs in subject_features:
            subject_data.append(np.concatenate(channel_coeffs))  # Concatenate all wavelet levels
        processed_signals.append(np.array(subject_data).reshape(-1, 1))
    return np.array(processed_signals)

eeg_data = prepare_input(dwt_features)

# Placeholder for labels (0: relaxed, 1: stressed)
labels = np.array([0 if i < len(eeg_data) // 2 else 1 for i in range(len(eeg_data))])

# Build CNN–BLSTM Model
model = Sequential([
    Conv1D(filters=95, kernel_size=2, activation='relu', input_shape=(eeg_data.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=47, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and Evaluate Model
X_train, X_test, y_train, y_test = train_test_split(eeg_data, labels, test_size=0.3, stratify=labels)

history = model.fit(X_train, y_train, epochs=100, batch_size=20, validation_split=0.3)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Performance Analysis with ROC Curve
y_pred_prob = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Convergence Curve Analysis (Training vs Validation Accuracy)
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Convergence Curve')
plt.show()

# Additional Performance Metrics (Confusion Matrix and Precision, Recall)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Predict classes
y_pred_classes = (y_pred_prob > 0.5).astype("int32")

# Confusion matrix and metrics
conf_matrix = confusion_matrix(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print("Confusion Matrix:\n", conf_matrix)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")