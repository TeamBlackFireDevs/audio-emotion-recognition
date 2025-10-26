# audio_emotion_recognition.py
# Python 3.9+ recommended
# pip install tensorflow librosa numpy scikit-learn matplotlib soundfile

import os
import re
import glob
import math
import random
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

# -----------------------------
# Config
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# >>>>>> EDIT THESE <<<<<<
DATASET_DIR  = "cremad_dataset"
DATASET_TYPE = "cremad"  # "ravdess" or "cremad"

# Audio & feature params
TARGET_SR        = 16000
DURATION_SEC     = 3.0      # pad/trim to this
SAMPLES_PER_CLIP = int(TARGET_SR * DURATION_SEC)

# Mel-spec params (CNN branch)
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 256

# MFCC params (LSTM branch)
N_MFCC = 40

# Training params
BATCH_SIZE   = 32
EPOCHS       = 30
VAL_SPLIT    = 0.15
TEST_SIZE    = 0.15
AUGMENT_P    = 0.4   # probability to apply an augmentation
SAVE_DIR     = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Utility: Label parsing
# -----------------------------
# RAVDESS filename example: 03-01-06-01-02-01-12.wav
# Field 3 (index 2) is emotion id: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised
RAVDESS_EMO_MAP = {
    "01":"neutral","02":"calm","03":"happy","04":"sad",
    "05":"angry","06":"fearful","07":"disgust","08":"surprised"
}

def parse_label_from_filename(path, dataset_type):
    fname = os.path.basename(path)
    if dataset_type.lower() == "ravdess":
        parts = fname.split(".")[0].split("-")
        if len(parts) >= 3:
            emo_code = parts[2]
            return RAVDESS_EMO_MAP.get(emo_code, None)
        return None
    elif dataset_type.lower() == "cremad":
        # CREMA-D example: 1001_DFA_ANG_XX.wav -> third token is emotion code
        # ANG, DIS, FEAR, HAPPY, NEUTRAL, SAD (variants exist)
        parts = fname.split(".")[0].split("_")
        if len(parts) >= 3:
            code = parts[2].upper()
            mapping = {
                "ANG": "angry",
                "DIS": "disgust",
                "FEA": "fearful",
                "FEAR":"fearful",
                "HAP": "happy",
                "NEU": "neutral",
                "SAD": "sad"
            }
            return mapping.get(code, None)
        return None
    else:
        raise ValueError("DATASET_TYPE must be 'ravdess' or 'cremad'")

# -----------------------------
# Audio loading + pad/trim
# -----------------------------
def load_audio_fixed(path, target_sr=TARGET_SR, target_len=SAMPLES_PER_CLIP):
    y, sr = librosa.load(path, sr=target_sr)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    else:
        y = y[:target_len]
    return y

# -----------------------------
# Augmentations
# -----------------------------
def augment_audio(y, sr):
    choice = random.choice(["noise","pitch","speed"])
    if choice == "noise":
        noise_amp = 0.005 * np.random.uniform() * np.amax(np.abs(y))
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    elif choice == "pitch":
        steps = np.random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        if len(y) < SAMPLES_PER_CLIP:
            y = np.pad(y, (0, SAMPLES_PER_CLIP - len(y)))
        else:
            y = y[:SAMPLES_PER_CLIP]
    else:  # speed
        rate = np.random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
        if len(y) < SAMPLES_PER_CLIP:
            y = np.pad(y, (0, SAMPLES_PER_CLIP - len(y)))
        else:
            y = y[:SAMPLES_PER_CLIP]
    return y

# -----------------------------
# Features: Mel-spectrogram & MFCC
# -----------------------------
def wav_to_melspec(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Standardize to zero mean/unit var per clip
    mu, sigma = S_db.mean(), S_db.std() + 1e-8
    S_db = (S_db - mu) / sigma
    return S_db

def wav_to_mfcc_seq(y, sr=TARGET_SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    # T x F (transpose to time-major)
    mfcc = mfcc.T
    # Standardize
    mu, sigma = mfcc.mean(), mfcc.std() + 1e-8
    mfcc = (mfcc - mu) / sigma
    return mfcc

def pad_or_trim_time_axis(feat, target_len):
    T = feat.shape[0]
    if T < target_len:
        pad = target_len - T
        return np.pad(feat, ((0, pad), (0, 0)), mode="constant")
    else:
        return feat[:target_len, :]

# -----------------------------
# Load dataset → arrays
# -----------------------------
def collect_files(dataset_dir):
    wavs = glob.glob(os.path.join(dataset_dir, "**", "*.wav"), recursive=True)
    return wavs

def build_dataset(dataset_dir, dataset_type, do_augment=True):
    X_mel, X_mfcc, y = [], [], []
    files = collect_files(dataset_dir)
    for path in files:
        label = parse_label_from_filename(path, dataset_type)
        if label is None:
            continue
        y_raw = load_audio_fixed(path)

        # maybe augment
        if do_augment and random.random() < AUGMENT_P:
            y_aug = augment_audio(y_raw, TARGET_SR)
        else:
            y_aug = y_raw

        mel = wav_to_melspec(y_aug)
        mfcc = wav_to_mfcc_seq(y_aug)

        # For CNN: need (H, W, 1). We'll set a fixed time length for consistency across clips
        # Compute expected time frames roughly = floor((N + hop)/hop); use mel shape for target
        # Standardize to a uniform width (e.g., 188 frames for 3s at hop=256 @ 16kHz ~ 188)
        TARGET_T_MEL = mel.shape[1]
        mel_cnn = mel  # (n_mels, T)
        mel_cnn = np.expand_dims(mel_cnn, -1)  # (n_mels, T, 1)

        # For LSTM: pad/truncate MFCC time steps to a fixed length (e.g., match mel T)
        TARGET_T_MFCC = TARGET_T_MEL
        mfcc_seq = pad_or_trim_time_axis(mfcc, TARGET_T_MFCC)  # (T, n_mfcc)

        X_mel.append(mel_cnn.astype(np.float32))
        X_mfcc.append(mfcc_seq.astype(np.float32))
        y.append(label)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    X_mel  = np.stack(X_mel, axis=0)
    X_mfcc = np.stack(X_mfcc, axis=0)
    return X_mel, X_mfcc, y_enc, class_names

print("Scanning & featurizing… (this can take a few minutes)")
X_mel, X_mfcc, y_all, class_names = build_dataset(DATASET_DIR, DATASET_TYPE, do_augment=True)
num_classes = len(class_names)
print(f"Samples: {len(y_all)} | Classes: {class_names}")

# Train/val/test split (stratified)
X_mel_train, X_mel_test, X_mfcc_train, X_mfcc_test, y_train, y_test = train_test_split(
    X_mel, X_mfcc, y_all, test_size=TEST_SIZE, random_state=SEED, stratify=y_all
)
X_mel_train, X_mel_val, X_mfcc_train, X_mfcc_val, y_train, y_val = train_test_split(
    X_mel_train, X_mfcc_train, y_train, test_size=VAL_SPLIT, random_state=SEED, stratify=y_train
)

# -----------------------------
# Models
# -----------------------------
def build_cnn_melspec(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="CNN_MelSpec")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_lstm_mfcc(time_steps, n_mfcc, num_classes):
    inputs = tf.keras.Input(shape=(time_steps, n_mfcc))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="LSTM_MFCC")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

cnn_input_shape = X_mel_train.shape[1:]   # (n_mels, T, 1)
time_steps      = X_mfcc_train.shape[1]   # T
mfcc_dim        = X_mfcc_train.shape[2]   # n_mfcc

model_cnn = build_cnn_melspec(cnn_input_shape, num_classes)
model_lstm = build_lstm_mfcc(time_steps, mfcc_dim, num_classes)

cb = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_cnn.keras"), monitor="val_accuracy", save_best_only=True)
]
cb2 = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(SAVE_DIR, "best_lstm.keras"), monitor="val_accuracy", save_best_only=True)
]

print("\nTraining CNN on Mel-spectrograms…")
hist_cnn = model_cnn.fit(
    X_mel_train, y_train,
    validation_data=(X_mel_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=1
)

print("\nTraining LSTM on MFCC sequences…")
hist_lstm = model_lstm.fit(
    X_mfcc_train, y_train,
    validation_data=(X_mfcc_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb2,
    verbose=1
)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_and_report(model, X_te, y_te, title):
    y_prob = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n=== {title} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=class_names))
    cm = confusion_matrix(y_te, y_pred)
    print("Confusion Matrix:\n", cm)

    # Simple matplotlib heatmap (no seaborn)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{title.replace(' ','_').lower()}_cm.png"), dpi=200)
    plt.close(fig)
    return acc

acc_cnn  = evaluate_and_report(model_cnn,  X_mel_test,  y_test, "CNN Mel-Spectrogram")
acc_lstm = evaluate_and_report(model_lstm, X_mfcc_test, y_test, "LSTM MFCC")

print(f"\nSummary → CNN: {acc_cnn:.4f} | LSTM: {acc_lstm:.4f}")
with open(os.path.join(SAVE_DIR, "results.txt"), "w") as f:
    f.write(f"Classes: {class_names}\n")
    f.write(f"CNN accuracy: {acc_cnn:.4f}\n")
    f.write(f"LSTM accuracy: {acc_lstm:.4f}\n")

# -----------------------------
# Quick inference demo
# -----------------------------
def predict_file(wav_path, model_type="cnn"):
    y = load_audio_fixed(wav_path)

    # CNN branch → mel-spectrogram
    mel = wav_to_melspec(y)
    mel = pad_or_trim_time_axis(mel.T, 188).T  # fix time axis to 188
    mel = np.expand_dims(mel, -1)[None, ...]   # (1, 128, 188, 1)

    # LSTM branch → MFCC
    mfcc = wav_to_mfcc_seq(y)
    mfcc = pad_or_trim_time_axis(mfcc, 188)    # fix to same time length
    mfcc = mfcc[None, ...]                      # (1, 188, 40)

    if model_type == "cnn":
        probs = model_cnn.predict(mel, verbose=0)[0]
    elif model_type == "lstm":
        probs = model_lstm.predict(mfcc, verbose=0)[0]
    else:
        raise ValueError("model_type must be 'cnn' or 'lstm'")

    return dict(zip(class_names, probs.tolist()))


# Example:
# demo = predict_file("/path/to/one/test.wav")
# print(demo)
