
# Experiment 2: Sigmoid Neuron for Audio Emotion Recognition
# Binary Classification: Positive vs Negative Emotions (same as Experiment 1)
# Created for Multi-Modal Emotion Recognition Project

import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update this path
TARGET_SR = 16000
DURATION_SEC = 3.0
SEED = 42
np.random.seed(SEED)

# Emotion mapping for binary classification
POSITIVE_EMOTIONS = ['happy', 'neutral']
NEGATIVE_EMOTIONS = ['angry', 'sad', 'fearful', 'disgust']

# -----------------------------
# Utility Functions from Original Project
# -----------------------------
def parse_cremad_label(filename):
    parts = filename.split('.')[0].split('_')
    if len(parts) >= 3:
        code = parts[2].upper()
        mapping = {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
            'FEAR': 'fearful', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        }
        return mapping.get(code, None)
    return None

def load_audio_fixed(path, target_sr=TARGET_SR):
    try:
        y, sr = librosa.load(path, sr=target_sr, duration=DURATION_SEC)
        target_len = int(target_sr * DURATION_SEC)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        return y
    except:
        return None

def extract_simple_features(y, sr=TARGET_SR):
    features = []
    features.append(np.mean(np.abs(y)))
    features.append(np.std(y))
    features.append(np.max(np.abs(y)))
    features.append(np.sqrt(np.mean(y**2)))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(spec_rolloff))
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(spec_bandwidth))
    return np.array(features)

# -----------------------------
# Sigmoid Neuron Implementation
# -----------------------------
class SigmoidNeuron:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0

        for epoch in range(self.max_epochs):
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)
            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(y_pred + 1e-7) + (1 - y) * np.log(1 - y_pred + 1e-7))
            self.losses.append(loss)
            # Gradient computation
            dW = np.dot((y_pred - y), X) / n_samples
            db = np.mean(y_pred - y)
            # Weights update
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            # Print every 100 epochs
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1} - Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# -----------------------------
# Data Loading and Processing
# -----------------------------
def load_dataset():
    X, y = [], []
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    processed_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        emotion = parse_cremad_label(filename)
        if emotion is None:
            continue
        if emotion in POSITIVE_EMOTIONS:
            binary_label = 1
        elif emotion in NEGATIVE_EMOTIONS:
            binary_label = 0
        else:
            continue
        audio = load_audio_fixed(file_path)
        if audio is not None:
            features = extract_simple_features(audio)
            X.append(features)
            y.append(binary_label)
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
    print(f"Total files processed: {processed_count}")
    return np.array(X), np.array(y)

# -----------------------------
# Evaluation and Visualization
# -----------------------------
def evaluate_model(model, X_test, y_test):
    probas = model.predict_proba(X_test)
    predictions = (probas >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probas)
    print(f"Sigmoid Neuron Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Negative', 'Positive']))
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    return accuracy, auc, predictions, cm

def plot_results(sigmoid_losses):
    plt.figure(figsize=(8,4))
    plt.plot(sigmoid_losses, 'g-', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.title('Sigmoid Neuron Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_2_sigmoid_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("=== Experiment 2: Sigmoid Neuron ===")
    print("Binary Classification: Positive vs Negative Emotions")
    X, y = load_dataset()
    if len(X) == 0:
        print("No data found! Please check DATASET_DIR path.")
        return
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution - Negative: {np.sum(y==0)}, Positive: {np.sum(y==1)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Training Sigmoid Neuron...")
    sigmoid_neuron = SigmoidNeuron(learning_rate=0.01, max_epochs=1000)
    sigmoid_neuron.fit(X_train_scaled, y_train)
    accuracy, auc, predictions, cm = evaluate_model(sigmoid_neuron, X_test_scaled, y_test)
    plot_results(sigmoid_neuron.losses)
    with open('experiment_2_results.txt', 'w') as f:
        f.write("=== Experiment 2: Sigmoid Neuron Results ===")
        f.write(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        f.write(f"Train/Test Split: {len(X_train)}/{len(X_test)}")
        f.write(f"Test Accuracy: {accuracy:.4f}")
        f.write(f"Test ROC-AUC: {auc:.4f}")
        f.write(f"Confusion Matrix: {cm}")
    return accuracy, auc

if __name__ == "__main__":
    main()
