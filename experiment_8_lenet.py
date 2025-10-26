# Experiment 8: LeNet-based Audio Emotion Recognition using Spectrogram Images
# Adapted from classic LeNet architecture for image classification
# Multi-Modal Emotion Recognition Project

import os
import numpy as np
import librosa
import glob
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update this path
TARGET_SR = 16000
DURATION_SEC = 3.0
N_MELS = 128
EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Feature Extraction
# -----------------------------
def parse_cremad_label(filename):
    parts = filename.split('.')[0].split('_')
    if len(parts) >= 3:
        code = parts[2].upper()
        mapping = {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'FEAR': 'fearful',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        }
        return mapping.get(code, None)
    return None

def load_audio_fixed(path, target_sr=TARGET_SR):
    y, sr = librosa.load(path, sr=target_sr, duration=DURATION_SEC)
    target_len = int(target_sr * DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        y = y[:target_len]
    return y

def audio_to_melspectrogram(y, sr=TARGET_SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)
    return S_norm

# -----------------------------
# Dataset Loading
# -----------------------------
def load_dataset():
    X, y = [], []
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    for file_path in files:
        emotion = parse_cremad_label(os.path.basename(file_path))
        if emotion not in EMOTIONS:
            continue
        audio = load_audio_fixed(file_path)
        mel_img = audio_to_melspectrogram(audio)
        X.append(mel_img)
        y.append(emotion)
    print(f"Total samples: {len(X)}")
    return np.array(X), np.array(y)

# -----------------------------
# LeNet Architecture
# -----------------------------
def build_lenet(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(6, kernel_size=(5,5), activation='tanh', padding='same'),
        AveragePooling2D(pool_size=(2,2)),
        Conv2D(16, kernel_size=(5,5), activation='tanh'),
        AveragePooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(120, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Training Pipeline
# -----------------------------
def main():
    print("=== Experiment 8: LeNet for Spectrogram Image Classification ===")
    X, y = load_dataset()
    if len(X) == 0:
        print("No dataset found.")
        return
    
    # Reshape and normalize for CNN input
    X = np.expand_dims(X, -1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded, num_classes=len(EMOTIONS))

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=SEED, stratify=y_encoded)

    model = build_lenet(input_shape=X_train.shape[1:], num_classes=len(EMOTIONS))
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=2)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Plot accuracy and loss
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LeNet Model Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LeNet Model Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiment_8_lenet_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
