
# Experiment 5: Linear Regression with Mini-Batch Stochastic Gradient Descent (SGD)
# Regression task on audio features

import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update this
TARGET_SR = 16000
DURATION_SEC = 3.0
SEED = 42
BATCH_SIZE = 32
np.random.seed(SEED)

# -----------------------------
# Utility functions
# -----------------------------
def parse_cremad_label(filename):
    parts = filename.split('.')[0].split('_')
    if len(parts) >= 3:
        code = parts[2].upper()
        mapping = {
            'ANG': 0, 'DIS': 1, 'FEA': 2, 'FEAR': 2,
            'HAP': 3, 'NEU': 4, 'SAD': 5
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
# Linear Regression with Mini-Batch SGD Implementation
# -----------------------------
class LinearRegressionMiniBatchSGD:
    def __init__(self, learning_rate=0.01, max_epochs=1000, batch_size=BATCH_SIZE):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for epoch in range(self.max_epochs):
            permuted_indices = np.random.permutation(n_samples)
            X_shuffled = X[permuted_indices]
            y_shuffled = y[permuted_indices]

            epoch_loss = 0
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred = np.dot(X_batch, self.weights) + self.bias
                error = y_pred - y_batch

                # Gradient
                dW = np.dot(error, X_batch) / len(y_batch)
                db = np.mean(error)

                # Update weights
                self.weights -= self.learning_rate * dW
                self.bias -= self.learning_rate * db

                batch_loss = np.mean(error ** 2)
                epoch_loss += batch_loss * len(y_batch)

            epoch_loss /= n_samples
            self.losses.append(epoch_loss)
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1} - MSE: {epoch_loss:.4f}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# -----------------------------
# Data Loading and Regression Target
# -----------------------------
def load_regression_dataset():
    X, y = [], []
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    processed_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        label = parse_cremad_label(filename)
        if label is None:
            continue
        audio = load_audio_fixed(file_path)
        if audio is not None:
            features = extract_simple_features(audio)
            X.append(features)
            y.append(label)  # numeric regression target
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
    print(f"Total files processed: {processed_count}")
    return np.array(X), np.array(y)

# -----------------------------
# Evaluation and Visualization
# -----------------------------
def plot_losses(losses):
    plt.figure(figsize=(8,4))
    plt.plot(losses, 'm-', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.title('Linear Regression with Mini-Batch SGD Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_5_lr_minibatch_sgd_loss.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=== Experiment 5: Linear Regression with Mini-Batch SGD ===")
    X, y = load_regression_dataset()
    if len(X) == 0:
        print("No data found! Please check DATASET_DIR path.")
        return
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegressionMiniBatchSGD(learning_rate=0.01, max_epochs=1000, batch_size=BATCH_SIZE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test R^2 Score: {r2:.4f}")

    plot_losses(model.losses)

    with open('experiment_5_lr_minibatch_sgd_results.txt', 'w') as f:
        f.write("=== Experiment 5: Linear Regression with Mini-Batch SGD Results ===")
        f.write(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        f.write(f"Train/Test Split: {len(X_train)}/{len(X_test)}")
        f.write(f"Test MSE: {mse:.4f}")
        f.write(f"Test R^2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()
