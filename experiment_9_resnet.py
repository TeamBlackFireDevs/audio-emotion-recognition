# Experiment 9 (PyTorch): ResNet18-based Audio Emotion Recognition using Spectrograms
# Multi-Modal Emotion Recognition Project (Fast GPU Version)

import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # path to your dataset
TARGET_SR = 16000
DURATION_SEC = 3.0
N_MELS = 128
BATCH_SIZE = 32
EPOCHS = 20
EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", DEVICE)

# -----------------------------
# Data Utilities
# -----------------------------
def parse_cremad_label(filename):
    parts = filename.split('.')[0].split('_')
    if len(parts) >= 3:
        mapping = {
            'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'FEAR': 'fearful',
            'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
        }
        return mapping.get(parts[2].upper(), None)
    return None

def load_audio_fixed(path):
    y, _ = librosa.load(path, sr=TARGET_SR, duration=DURATION_SEC)
    target_len = int(TARGET_SR * DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def audio_to_melspectrogram(y):
    S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)
    return S_norm

# -----------------------------
# Custom Dataset
# -----------------------------
class SpectrogramDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = load_audio_fixed(self.files[idx])
        mel = audio_to_melspectrogram(audio)
        mel = np.stack([mel, mel, mel], axis=0)  # 3-channel for ResNet
        if self.transform:
            mel = self.transform(torch.tensor(mel, dtype=torch.float32))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

# -----------------------------
# Dataset Loading
# -----------------------------
def load_dataset():
    X, y = [], []
    files = glob.glob(os.path.join(DATASET_DIR, '**', '*.wav'), recursive=True)
    for f in files:
        emotion = parse_cremad_label(os.path.basename(f))
        if emotion not in EMOTIONS:
            continue
        X.append(f)
        y.append(emotion)
    print(f"Loaded {len(X)} samples")
    return np.array(X), np.array(y)

# -----------------------------
# Model Setup
# -----------------------------
def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# -----------------------------
# Training & Evaluation
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        train_loss = 0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Loss={train_loss/len(train_loader):.3f}")

    return train_accs, val_accs

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# -----------------------------
# Main
# -----------------------------
def main():
    X, y = load_dataset()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    transform = transforms.Resize((128, 128))
    train_data = SpectrogramDataset(X_train, y_train, transform)
    test_data = SpectrogramDataset(X_test, y_test, transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(len(EMOTIONS))
    train_accs, val_accs = train_model(model, train_loader, test_loader)

    # Final evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            preds = outputs.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Plot
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiment_9_resnet18_torch_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
