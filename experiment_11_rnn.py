# Experiment 11: Character-Level Language Modeling using RNN (CREMA-D adaptation)
# Multi-Modal Emotion Recognition Project

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import glob

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update this path
SEED = 42
BATCH_SIZE = 32
HIDDEN_SIZE = 128
EMBEDDING_DIM = 64
EPOCHS = 30
LR = 0.003
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Data Preparation from CREMA-D filenames
# -----------------------------
def get_cremad_text_sequences(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, '**', '*.wav'), recursive=True)
    filenames = [os.path.basename(f).split('.')[0] for f in files]
    text_corpus = " ".join(filenames)
    return text_corpus

text_data = get_cremad_text_sequences(DATASET_DIR)
print(f"Extracted corpus length: {len(text_data)} characters")

# Create vocabulary
chars = sorted(list(set(text_data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Encode and decode functions
def encode(s): return [stoi[c] for c in s]
def decode(indices): return ''.join([itos[i] for i in indices])

# Create sequence dataset
class CharDataset(Dataset):
    def __init__(self, data, context_length=25):
        self.data = torch.tensor(encode(data), dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.context_length + 1]
        return chunk[:-1], chunk[1:]

dataset = CharDataset(text_data, context_length=25)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# RNN Model
# -----------------------------
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNNLanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden

model = RNNLanguageModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop
# -----------------------------
losses = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# -----------------------------
# Text Generation to Test Learning
# -----------------------------
def generate_text(model, start_sequence='100', length=100):
    model.eval()
    context = torch.tensor(encode(start_sequence), dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = list(start_sequence)
    hidden = None
    for _ in range(length):
        logits, hidden = model(context, hidden)
        probs = torch.softmax(logits[:, -1, :], dim=-1).cpu().detach().numpy().ravel()
        next_char_idx = np.random.choice(len(probs), p=probs)
        next_char = itos[next_char_idx]
        generated.append(next_char)
        context = torch.tensor([[next_char_idx]], dtype=torch.long).to(DEVICE)
    return ''.join(generated)

sample_output = generate_text(model, start_sequence='1001_', length=120)
print("\nGenerated sequence sample:")
print(sample_output)

# -----------------------------
# Plot Training Loss
# -----------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(losses, color='purple', marker='o')
plt.title('RNN Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('experiment_11_rnn_language_model_loss.png', dpi=300)
plt.show()
