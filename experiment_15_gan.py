# Experiment 15: GAN for Spectrogram Image Generation (Multi-Modal Emotion Recognition)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import glob
import matplotlib.pyplot as plt

# Configuration
DATASET_DIR = "cremad_dataset"
TARGET_SR = 16000
DURATION_SEC = 3.0
N_MELS = 128
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.0002
BETAS = (0.5, 0.999)
IMG_SIZE = 64  # resize mel-spectrograms to 64x64 for DCGAN
Z_DIM = 100  # latent noise vector size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preparation
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
        y = np.pad(y, (0, target_len - len(y)), 'constant')
    else:
        y = y[:target_len]
    return y

def audio_to_melspectrogram_img(y):
    S = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    img = np.array(Image.fromarray((S_norm * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE)))
    return img / 255.0  # Normalize to [0,1]

class SpecDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        y = load_audio_fixed(self.files[idx])
        spec_img = audio_to_melspectrogram_img(y)
        spec_img = np.expand_dims(spec_img, axis=0)  # 1 channel
        return torch.tensor(spec_img, dtype=torch.float32)

def load_dataset():
    files = []
    for f in glob.glob(os.path.join(DATASET_DIR, '**','*.wav'), recursive=True):
        if parse_cremad_label(os.path.basename(f)) is not None:
            files.append(f)
    return files

# Generator Model
class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super().__init__()
        self.gen = nn.Sequential(
            # input Z goes into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.gen(input)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        out = self.dis(input)
        return out.mean(dim=[1, 2, 3])  # shape -> [batch_size]

# -----------------------------
# Training Loop
# -----------------------------
def train_dcgan(generator, discriminator, dataloader, epochs):
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    optimizerD = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=DEVICE)

    gen_losses, dis_losses = [], []

    for epoch in range(epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)

            # Train discriminator
            discriminator.zero_grad()
            label = torch.full((batch_size,), 1.0, device=DEVICE, dtype=torch.float)
            output = discriminator(real_images)
            lossD_real = criterion(output, label)
            lossD_real.backward()

            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_images = generator(noise)
            label.fill_(0.0)
            output = discriminator(fake_images.detach())
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # Train generator
            generator.zero_grad()
            label.fill_(1.0)
            output = discriminator(fake_images)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()


        gen_losses.append(lossG.item())
        dis_losses.append(lossD.item())

        print(f"Epoch [{epoch+1}/{epochs}], Generator Loss: {lossG.item():.4f}, Discriminator Loss: {lossD.item():.4f}")

        # Save generated images periodically (code omitted for brevity)

    # Plot Losses
    plt.figure(figsize=(8,6))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(dis_losses, label='Discriminator Loss')
    plt.title('DCGAN Training Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiment_15_dcgan_losses.png', dpi=300)
    plt.show()

# -----------------------------
# Main
# -----------------------------
from PIL import Image

def main():
    print("=== Experiment 15: GAN for Generating Mel-Spectrograms ===")
    files = load_dataset()
    dataset = SpecDataset(files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(Z_DIM, 64).to(DEVICE)
    discriminator = Discriminator(64).to(DEVICE)

    train_dcgan(generator, discriminator, dataloader, EPOCHS)

if __name__ == "__main__":
    main()
