# experiment_14_rbm_fixed.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"
TARGET_SR = 16000
DURATION_SEC = 3.0
N_MELS = 128
BATCH_SIZE = 32
EPOCHS = 20
HIDDEN_UNITS = 256
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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

def audio_to_melspectrogram_flat(y, sr=TARGET_SR, n_mels=N_MELS, hop_length=512, eps=1e-8):
    """
    Returns a flattened, normalized mel-spectrogram in range [0,1], with NaNs handled.
    """
    # compute mel spectrogram (shape: n_mels x frames)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)  # in dB
    min_val = float(np.nanmin(S_db))
    max_val = float(np.nanmax(S_db))
    denom = (max_val - min_val)
    if denom <= 0:
        # all values equal (rare) -> return zeros
        S_norm = np.zeros_like(S_db, dtype=np.float32)
    else:
        S_norm = (S_db - min_val) / (denom + eps)
        # fix any possible numeric issues
        S_norm = np.nan_to_num(S_norm, posinf=0.0, neginf=0.0)
        S_norm = np.clip(S_norm, 0.0, 1.0)
    return S_norm.flatten().astype(np.float32)

class SpectrogramDataset(Dataset):
    def __init__(self, files, hop_length=512):
        self.files = files
        self.hop_length = hop_length
        self._cache = {}
        # Pre-cache spectrograms (optional but helpful)
        for f in files:
            y = load_audio_fixed(f)
            flat = audio_to_melspectrogram_flat(y, hop_length=self.hop_length)
            self._cache[f] = flat

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        spec_flat = self._cache[f]
        return torch.from_numpy(spec_flat).float()

def load_dataset_files():
    files = []
    for f in glob.glob(os.path.join(DATASET_DIR, '**', '*.wav'), recursive=True):
        files.append(f)
    return sorted(files)

# -----------------------------
# RBM Model
# -----------------------------
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super().__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        # small initialization
        self.W = nn.Parameter(torch.randn(hidden_units, visible_units, dtype=torch.float32) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units, dtype=torch.float32))
        self.v_bias = nn.Parameter(torch.zeros(visible_units, dtype=torch.float32))

    def sample_h(self, v):
        """
        Compute probability of hidden given visible. Return probabilities and (optional) bernoulli samples.
        We will mostly use probabilities for reconstruction and loss to keep gradients stable.
        """
        wx = torch.matmul(v, self.W.t()) + self.h_bias  # [batch, hidden_units]
        p_h_given_v = torch.sigmoid(wx)
        # sampling is optional; do not sample when computing gradients to avoid stochastic asserts
        # sample_h = torch.bernoulli(p_h_given_v)
        return p_h_given_v  # return probabilities only

    def sample_v(self, h):
        """
        Compute probabilities for visible units given hidden.
        Return probabilities (in [0,1]) — do not sample for reconstruction loss.
        """
        wx = torch.matmul(h, self.W) + self.v_bias  # [batch, visible_units]
        p_v_given_h = torch.sigmoid(wx)
        return p_v_given_h

    def forward(self, v):
        """
        One-step reconstruction: v -> p(h|v) -> p(v'|h)
        Return reconstructed probabilities (clamped to [0,1]).
        """
        p_h = self.sample_h(v)             # [batch, hidden]
        p_v = self.sample_v(p_h)           # [batch, visible]
        p_v = torch.clamp(p_v, 0.0, 1.0)
        return p_v

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W.t()) + self.h_bias
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return -vbias_term - hidden_term

# -----------------------------
# Training
# -----------------------------
def train_rbm(rbm, dataloader, lr, epochs):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    mse = nn.MSELoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        nb = 0
        for batch in dataloader:
            batch = batch.to(DEVICE)
            # ensure inputs are in [0,1] and finite
            batch = torch.clamp(batch, 0.0, 1.0)
            # forward (probabilistic recon)
            optimizer.zero_grad()
            v_recon = rbm(batch)
            loss = mse(batch, v_recon)  # compare probabilities
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            nb += 1
        avg_loss = epoch_loss / max(1, nb)
        print(f"Epoch {epoch+1}/{epochs}, Reconstruction Loss: {avg_loss:.6f}")

# -----------------------------
# Embedding Visualization
# -----------------------------
def visualize_embeddings(rbm, dataloader, max_points=1000):
    rbm.eval()
    all_reps = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(DEVICE)
            batch = torch.clamp(batch, 0.0, 1.0)
            p_h = rbm.sample_h(batch)  # probabilities [batch, hidden]
            all_reps.append(p_h.cpu().numpy())
            # optional: limit total points for TSNE speed
            if sum(arr.shape[0] for arr in all_reps) >= max_points:
                break
    all_reps = np.vstack(all_reps)
    # reduce size if too many points
    if all_reps.shape[0] > max_points:
        idx = np.random.choice(all_reps.shape[0], max_points, replace=False)
        all_reps = all_reps[idx]

    tsne = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(all_reps)
    kmeans = KMeans(n_clusters=6, random_state=SEED).fit(reduced)
    plt.figure(figsize=(8,6))
    for cluster_idx in range(6):
        pts = reduced[kmeans.labels_ == cluster_idx]
        if pts.size == 0:
            continue
        plt.scatter(pts[:, 0], pts[:, 1], label=f'Cluster {cluster_idx}', s=8, alpha=0.7)
    plt.legend(markerscale=2)
    plt.title("t-SNE Visualization of RBM Hidden Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    out_file = 'experiment_14_rbm_embedding.png'
    plt.savefig(out_file, dpi=300)
    print(f"Saved embedding plot to: {out_file}")
    plt.show()

# -----------------------------
# Main
# -----------------------------
def main():
    print("=== Experiment 14: Scene Understanding using RBMs (fixed) ===")
    files = load_dataset_files()
    if len(files) == 0:
        print("❌ No audio files found in", DATASET_DIR)
        return

    print(f"✅ Found {len(files)} audio files")

    dataset = SpectrogramDataset(files)
    sample_vec = dataset[0].numpy()
    visible_dim = sample_vec.size
    print("✅ Detected visible dimension:", visible_dim)

    # Use num_workers=0 for Windows stability
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == 'cuda')
    )

    rbm = RBM(visible_units=visible_dim, hidden_units=HIDDEN_UNITS).to(DEVICE)
    print("✅ Model initialized")

    # Debug: ensure DataLoader works
    try:
        first_batch = next(iter(dataloader))
        print("✅ DataLoader working, first batch shape:", first_batch.shape)
    except Exception as e:
        print("❌ DataLoader failed:", e)
        return

    print("🚀 Starting training...")
    train_rbm(rbm, dataloader, LR, EPOCHS)
    print("📊 Training finished. Generating embeddings...")
    visualize_embeddings(rbm, dataloader, max_points=1000)


if __name__ == "__main__":
    main()
