# Experiment 10: Transfer Learning using VGG16 for Audio Spectrogram Classification
# Optimized PyTorch Implementation with Precomputed Mel-Spectrograms

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"
CACHE_DIR = "mel_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TARGET_SR = 16000
DURATION_SEC = 3.0
N_MELS = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

# -----------------------------
# Data Preparation
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

def load_audio_fixed(path, target_sr=TARGET_SR):
    y, _ = librosa.load(path, sr=target_sr, duration=DURATION_SEC)
    target_len = int(target_sr * DURATION_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), 'constant')
    else:
        y = y[:target_len]
    return y

def audio_to_melspectrogram(y, sr=TARGET_SR, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - np.min(S_db)) / (np.max(S_db) - np.min(S_db))
    return S_norm.astype(np.float32)

def precompute_cache(files):
    cached = []
    for f in tqdm(files, desc="Precomputing mels"):
        base = os.path.splitext(os.path.basename(f))[0]
        npy_path = os.path.join(CACHE_DIR, base + ".npy")
        if not os.path.exists(npy_path):
            y = load_audio_fixed(f, target_sr=TARGET_SR)
            mel = audio_to_melspectrogram(y, TARGET_SR)
            np.save(npy_path, mel)
        cached.append(npy_path)
    return cached

class CachedSpectrogramDataset(Dataset):
    def __init__(self, mel_paths, labels, transform=None):
        self.mel_paths = mel_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.mel_paths)
    def __getitem__(self, idx):
        mel = np.load(self.mel_paths[idx])  # (n_mels, time)
        mel_img = np.stack([mel, mel, mel], axis=-1)  # 3-channel
        label = int(self.labels[idx])
        if self.transform:
            mel_img = self.transform(mel_img)
        return mel_img, torch.tensor(label, dtype=torch.long)

# -----------------------------
# Load Dataset
# -----------------------------
def load_dataset():
    files, labels = [], []
    for f in glob.glob(os.path.join(DATASET_DIR, '**', '*.wav'), recursive=True):
        emotion = parse_cremad_label(os.path.basename(f))
        if emotion in EMOTIONS:
            files.append(f)
            labels.append(emotion)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print("Class counts:", Counter(labels))
    return files, labels, le.classes_

# -----------------------------
# Model Definition (VGG16 Transfer Learning)
# -----------------------------
def build_vgg16_model(num_classes=6):
    model = models.vgg16(weights="IMAGENET1K_V1")
    for param in model.features.parameters():
        param.requires_grad = False  # freeze conv base
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model.to(DEVICE)

# -----------------------------
# Training & Evaluation
# -----------------------------
def train_model(model, dataloader, criterion, optimizer, scaler=None):
    model.train()
    total_loss, correct = 0.0, 0
    for inputs, labels in tqdm(dataloader, leave=False):
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), correct / len(dataloader.dataset), all_true, all_preds

# -----------------------------
# Main Function
# -----------------------------
def main():
    print("=== Experiment 10: VGG16 Transfer Learning for Audio Spectrogram Classification ===")
    files, labels, classes = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    X_train_cached = precompute_cache(X_train)
    X_test_cached = precompute_cache(X_test)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = CachedSpectrogramDataset(X_train_cached, y_train, transform)
    test_data = CachedSpectrogramDataset(X_test_cached, y_test, transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_vgg16_model(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier[6].parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    train_losses, test_losses, accuracies = [], [], []

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_model(model, train_loader, criterion, optimizer, scaler)
        te_loss, te_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        accuracies.append(te_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} => TrainAcc: {tr_acc:.3f} | ValAcc: {te_acc:.3f}")

        # Optional: unfreeze last conv block after 3 epochs
        if epoch == 2:
            for p in model.features[24:].parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.classifier[6].parameters(), 'lr': LR},
                {'params': model.features[24:].parameters(), 'lr': 1e-5}
            ], weight_decay=1e-4)

    print("\nFinal Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Plot Accuracy and Loss

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiment_10_vgg16_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()