
# Experiment 6: Optimizing Neural Networks with L2 Regularization, Dropout, Data Augmentation & Early Stopping
# Multi-class classification on audio emotions using Keras/TensorFlow

import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update this
TARGET_SR = 16000
DURATION_SEC = 3.0
SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
np.random.seed(SEED)
tf.random.set_seed(SEED)

EMOTION_CLASSES = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

# -----------------------------
# Data Augmentation - Audio Manipulations
# -----------------------------
def augment_audio(y, sr):
    # Random noise addition
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amp * np.random.normal(size=y.shape[0])

    # Random pitch shift
    n_steps = np.random.uniform(-2, 2)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # Random time stretch
    rate = np.random.uniform(0.8, 1.2)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)

    # Choose one augmentation randomly
    augments = [y_noise, y_pitch, y_stretch]
    y_aug = augments[np.random.randint(0, len(augments))]

    # If augmented audio is not same length, pad or trim
    target_len = int(sr * DURATION_SEC)
    if len(y_aug) < target_len:
        y_aug = np.pad(y_aug, (0, target_len - len(y_aug)), mode='constant')
    else:
        y_aug = y_aug[:target_len]

    return y_aug

# -----------------------------
# Utility functions
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
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(spec_rolloff))
    features.append(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean())
    return np.array(features)

# -----------------------------
# Data Loading and Augmentation
# -----------------------------
def load_dataset():
    X, y = [], []
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    processed_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        emotion = parse_cremad_label(filename)
        if emotion not in EMOTION_CLASSES:
            continue

        # Original audio
        audio = load_audio_fixed(file_path)
        if audio is not None:
            X.append(extract_simple_features(audio))
            y.append(emotion)
            processed_count += 1

            # Augmented audio data
            aug_audio = augment_audio(audio, TARGET_SR)
            X.append(extract_simple_features(aug_audio))
            y.append(emotion)
            processed_count += 1

        if processed_count % 100 == 0:
            print(f"Processed {processed_count} samples (including augmentation)...")
    print(f"Total samples processed: {processed_count}")
    return np.array(X), np.array(y)

# -----------------------------
# Model Construction
# -----------------------------
def build_model(input_dim, num_classes, l2_lambda=0.01, dropout_rate=0.3):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate * 0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Training and Evaluation
# -----------------------------
def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('experiment_6_nn_optimization_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=== Experiment 6: NN Optimization (L2, Dropout, Augmentation, Early Stopping) ===")
    X, y = load_dataset()
    if len(X) == 0:
        print("No data found! Please check DATASET_DIR path.")
        return
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=SEED, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model(input_dim=X_train_scaled.shape[1], num_classes=len(EMOTION_CLASSES))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2
    )

    plot_history(history)

    print("Evaluating on test data")
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_CLASSES))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    with open('experiment_6_nn_optimized_results.txt', 'w') as f:
        f.write("=== Experiment 6: NN Optimization Results ===")
        f.write(f"Dataset samples: {len(X)}")
        f.write(f"Train/Test Split: {len(X_train)}/{len(X_test)}")
        f.write(f"Test Accuracy: {test_acc:.4f}")
        f.write("Classification Report:")
        f.write(classification_report(y_true, y_pred, target_names=EMOTION_CLASSES))
        f.write("Confusion Matrix:")
        f.write(str(cm))

if __name__ == "__main__":
    main()
