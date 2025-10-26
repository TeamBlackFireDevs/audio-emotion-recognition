# experiment13_fixed_with_graphs.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update path
TARGET_SR = 16000
DURATION_SEC = 3.0
N_MFCC = 13
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
SEED = 42
MAX_SEQ_LEN = 75
HIDDEN_DIM = 64
EMBED_DIM = 32

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Dataset Prep
# -----------------------------
def parse_cremad_label(filename):
    parts = filename.split('.')[0].split('_')
    if len(parts) >= 3:
        mapping = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fearful',
            'FEAR': 'fearful',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
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

def extract_mfcc_seq(y):
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return mfcc.T

def load_dataset_files_and_labels():
    files, labels = [], []
    for f in glob.glob(os.path.join(DATASET_DIR, '**', '*.wav'), recursive=True):
        emotion = parse_cremad_label(os.path.basename(f))
        if emotion and emotion in EMOTIONS:
            files.append(f)
            labels.append(emotion)
    return files, labels

# -----------------------------
# Dataset with MFCC caching
# -----------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, file_paths, labels, max_seq_len=MAX_SEQ_LEN):
        self.file_paths = file_paths
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
        self.cache = {}

        for f in file_paths:
            y = load_audio_fixed(f)
            mfcc = extract_mfcc_seq(y)
            if len(mfcc) < max_seq_len:
                pad_width = ((0, max_seq_len - len(mfcc)), (0, 0))
                mfcc = np.pad(mfcc, pad_width)
            else:
                mfcc = mfcc[:max_seq_len]
            self.cache[f] = mfcc.astype(np.float32)

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        f = self.file_paths[idx]
        src_seq = self.cache[f]
        target_idx = self.emotion_to_idx[self.labels[idx]]
        tgt_seq = np.full(self.max_seq_len, target_idx, dtype=np.int64)
        return torch.tensor(src_seq), torch.tensor(tgt_seq)

# -----------------------------
# Model
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        h = torch.tanh(self.fc_h(torch.cat((h[-2], h[-1]), dim=1))).unsqueeze(0)
        c = torch.tanh(self.fc_c(torch.cat((c[-2], c[-1]), dim=1))).unsqueeze(0)
        return out, h, c

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_out):
        src_len = enc_out.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_out), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2, output_dim)
        self.attn = Attention(hidden_dim)

    def forward(self, input, hidden, cell, enc_out):
        input = input.unsqueeze(1)
        emb = self.embedding(input)
        attn_weights = self.attn(hidden[-1], enc_out).unsqueeze(1)
        context = torch.bmm(attn_weights, enc_out)
        lstm_in = torch.cat((emb, context), dim=2)
        out, (h, c) = self.lstm(lstm_in, (hidden, cell))
        out = out.squeeze(1)
        context = context.squeeze(1)
        pred = self.fc_out(torch.cat((out, context), dim=1))
        return pred, h, c

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, device):
        super().__init__()
        self.enc, self.dec, self.device = enc, dec, device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch, trg_len = trg.size()
        vocab = self.dec.fc_out.out_features
        out = torch.zeros(batch, trg_len, vocab).to(self.device)
        enc_out, h, c = self.enc(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            pred, h, c = self.dec(input, h, c, enc_out)
            out[:, t] = pred
            teacher_force = np.random.rand() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return out

# -----------------------------
# Training
# -----------------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for src, trg in loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        opt.zero_grad()
        out = model(src, trg)
        loss = loss_fn(out[:, 1:].reshape(-1, out.size(-1)), trg[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader, loss_fn):
    model.eval()
    total = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            out = model(src, trg, 0)
            loss = loss_fn(out[:, 1:].reshape(-1, out.size(-1)), trg[:, 1:].reshape(-1))
            total += loss.item()
    return total / len(loader)

# -----------------------------
# Main
# -----------------------------
def main():
    print("=== Experiment 13 (Fixed & Fast with Graphs) ===")
    files, labels = load_dataset_files_and_labels()
    if not files:
        print("❌ No dataset found.")
        return

    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=SEED)
    train_ds, test_ds = Seq2SeqDataset(X_train, y_train), Seq2SeqDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)

    enc = Encoder(N_MFCC, HIDDEN_DIM)
    dec = Decoder(len(EMOTIONS), EMBED_DIM, HIDDEN_DIM)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_dl, opt, loss_fn)
        val_loss = eval_epoch(model, test_dl, loss_fn)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{EPOCHS} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")

    # -----------------------------
    # Plot Loss Graph
    # -----------------------------
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('experiment_13_neural_machine_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
