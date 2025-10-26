
# Experiment 1: MP Neuron and Perceptron for Audio Emotion Recognition
# Binary Classification: Positive vs Negative Emotions
# Created for Multi-Modal Emotion Recognition Project

import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
# Positive: happy, neutral
# Negative: angry, sad, fearful, disgust
POSITIVE_EMOTIONS = ['happy', 'neutral']
NEGATIVE_EMOTIONS = ['angry', 'sad', 'fearful', 'disgust']

# -----------------------------
# Utility Functions from Original Project
# -----------------------------
def parse_cremad_label(filename):
    """Extract emotion from CREMA-D filename format: 1001_DFA_ANG_XX.wav"""
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
    """Load and standardize audio file"""
    try:
        y, sr = librosa.load(path, sr=target_sr, duration=DURATION_SEC)
        # Pad if too short
        target_len = int(target_sr * DURATION_SEC)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        else:
            y = y[:target_len]
        return y
    except:
        return None

def extract_simple_features(y, sr=TARGET_SR):
    """Extract simple statistical features suitable for MP Neuron/Perceptron"""
    features = []

    # Time-domain features
    features.append(np.mean(np.abs(y)))           # Mean amplitude
    features.append(np.std(y))                    # Standard deviation
    features.append(np.max(np.abs(y)))            # Peak amplitude
    features.append(np.sqrt(np.mean(y**2)))       # RMS energy

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Spectral features
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Spectral centroid (brightness)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(spec_rolloff))

    # Spectral bandwidth
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(spec_bandwidth))

    return np.array(features)

# -----------------------------
# MP Neuron Implementation
# -----------------------------
class MPNeuron:
    def __init__(self):
        self.threshold = 0
        self.feature_count = 0

    def fit(self, X, y):
        """Train MP Neuron by finding optimal threshold"""
        # Binarize features (1 if above median, 0 otherwise)
        X_binary = self._binarize_features(X)
        self.feature_count = X_binary.shape[1]

        best_accuracy = 0
        best_threshold = 0

        # Try different thresholds
        for threshold in range(self.feature_count + 1):
            predictions = self._predict_with_threshold(X_binary, threshold)
            accuracy = accuracy_score(y, predictions)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold
        self.best_train_accuracy = best_accuracy

        print(f"MP Neuron - Best threshold: {self.threshold}, Training accuracy: {best_accuracy:.4f}")

    def _binarize_features(self, X):
        """Convert continuous features to binary"""
        # Use median as threshold for each feature
        medians = np.median(X, axis=0)
        return (X > medians).astype(int)

    def _predict_with_threshold(self, X_binary, threshold):
        """Predict using sum of inputs vs threshold"""
        sums = np.sum(X_binary, axis=1)
        return (sums >= threshold).astype(int)

    def predict(self, X):
        """Make predictions on new data"""
        X_binary = self._binarize_features(X)
        return self._predict_with_threshold(X_binary, self.threshold)

# -----------------------------
# Perceptron Implementation
# -----------------------------
class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_errors = []

    def fit(self, X, y):
        """Train perceptron using iterative weight updates"""
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0

        # Convert labels to -1, +1 for perceptron learning
        y_perceptron = np.where(y == 0, -1, 1)

        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                # Calculate prediction
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = 1 if linear_output >= 0 else -1

                # Update weights if prediction is wrong
                if prediction != y_perceptron[i]:
                    self.weights += self.learning_rate * y_perceptron[i] * X[i]
                    self.bias += self.learning_rate * y_perceptron[i]
                    errors += 1

            self.training_errors.append(errors)

            # Stop if no errors (linearly separable)
            if errors == 0:
                print(f"Perceptron converged after {epoch + 1} epochs")
                break

        if errors > 0:
            print(f"Perceptron did not converge after {self.max_epochs} epochs")

    def predict(self, X):
        """Make predictions on new data"""
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0).astype(int)

# -----------------------------
# Data Loading and Processing
# -----------------------------
def load_dataset():
    """Load and process audio files for binary classification"""
    print("Loading dataset...")

    X, y = [], []
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)

    processed_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        emotion = parse_cremad_label(filename)

        if emotion is None:
            continue

        # Convert to binary classification
        if emotion in POSITIVE_EMOTIONS:
            binary_label = 1  # Positive
        elif emotion in NEGATIVE_EMOTIONS:
            binary_label = 0  # Negative
        else:
            continue  # Skip unknown emotions

        # Load audio and extract features
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
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n=== {model_name} Results ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                              target_names=['Negative', 'Positive']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, predictions, cm

def plot_results(mp_accuracy, perc_accuracy, perc_errors):
    """Create visualization of results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    models = ['MP Neuron', 'Perceptron']
    accuracies = [mp_accuracy, perc_accuracy]

    bars = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Comparison: MP Neuron vs Perceptron')
    ax1.set_ylim(0, 1)

    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    # Perceptron learning curve
    ax2.plot(perc_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Number of Errors')
    ax2.set_title('Perceptron Learning Curve')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_1_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("=== Experiment 1: MP Neuron and Perceptron ===")
    print("Binary Classification: Positive vs Negative Emotions\n")

    # Load data
    X, y = load_dataset()

    if len(X) == 0:
        print("No data found! Please check DATASET_DIR path.")
        return

    print(f"Dataset loaded: {len(X)} samples")
    print(f"Feature vector size: {X.shape[1]}")
    print(f"Class distribution - Negative: {np.sum(y==0)}, Positive: {np.sum(y==1)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Standardize features for Perceptron
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate MP Neuron
    print("\n" + "="*50)
    print("Training MP Neuron...")
    mp_neuron = MPNeuron()
    mp_neuron.fit(X_train, y_train)
    mp_accuracy, mp_pred, mp_cm = evaluate_model(mp_neuron, X_test, y_test, "MP Neuron")

    # Train and evaluate Perceptron
    print("\n" + "="*50)
    print("Training Perceptron...")
    perceptron = Perceptron(learning_rate=0.01, max_epochs=1000)
    perceptron.fit(X_train_scaled, y_train)
    perc_accuracy, perc_pred, perc_cm = evaluate_model(perceptron, X_test_scaled, y_test, "Perceptron")

    # Create visualization
    plot_results(mp_accuracy, perc_accuracy, perceptron.training_errors)

    # Save results
    with open('experiment_1_results.txt', 'w') as f:
        f.write("=== Experiment 1: MP Neuron and Perceptron Results ===\n\n")
        f.write(f"Dataset: {len(X)} samples, {X.shape[1]} features\n")
        f.write(f"Train/Test Split: {len(X_train)}/{len(X_test)}\n\n")
        f.write(f"MP Neuron Accuracy: {mp_accuracy:.4f}\n")
        f.write(f"Perceptron Accuracy: {perc_accuracy:.4f}\n\n")
        f.write(f"MP Neuron Threshold: {mp_neuron.threshold}\n")
        f.write(f"Perceptron Converged: {perceptron.training_errors[-1] == 0}\n")

    return mp_accuracy, perc_accuracy

if __name__ == "__main__":
    main()
