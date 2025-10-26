
# Experiment 7: Skip-Gram Model Adapted for Audio Features
# Instead of predicting words in context, predict audio feature frames from context frames
# Multi-Modal Emotion Recognition Project

import os
import numpy as np
import librosa
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Dot, Reshape
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "cremad_dataset"  # Update this path
TARGET_SR = 16000
DURATION_SEC = 3.0
SEED = 42
CONTEXT_WINDOW = 2  # number of frames before and after to predict
EMBEDDING_DIM = 64
BATCH_SIZE = 128
EPOCHS = 30

np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Utility Functions
# -----------------------------
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


def audio_to_mfcc_sequence(y, sr=TARGET_SR, n_mfcc=40, hop_length=512):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = mfcc.T  # Shape: (frames, n_mfcc)
    return mfcc

# -----------------------------
# Generate skip-gram training pairs from audio MFCC frames
# -----------------------------
def generate_skipgram_pairs(mfcc_seq, window=CONTEXT_WINDOW):
    pairs = []
    frame_count = len(mfcc_seq)
    for i in range(frame_count):
        center = mfcc_seq[i]
        context_indices = list(range(max(0, i - window), i)) + list(range(i + 1, min(frame_count, i + window + 1)))
        for ctx_idx in context_indices:
            context = mfcc_seq[ctx_idx]
            # Append (center, context) pair
            pairs.append((center, context))
    return pairs

# -----------------------------
# Build and train a simple skip-gram model on MFCC
# Using dense embeddings to predict context frames from center frames
# -----------------------------
class SkipGramTrainer:
    def __init__(self, embedding_dim=EMBEDDING_DIM, n_mfcc=40):
        self.embedding_dim = embedding_dim
        self.n_mfcc = n_mfcc
        self.model = None

    def build_model(self):
        # Input center frame
        center_input = Input(shape=(self.n_mfcc,), name='center_frame')
        # Embedding (dense representation)
        x = Dense(self.embedding_dim, activation='relu')(center_input)
        # Output is reconstruction (prediction) of context frame
        output = Dense(self.n_mfcc, activation='linear')(x)
        self.model = Model(inputs=center_input, outputs=output)
        self.model.compile(optimizer=Adam(0.001), loss='mse')

    def train(self, pairs, batch_size=BATCH_SIZE, epochs=EPOCHS):
        centers = np.array([pair[0] for pair in pairs])
        contexts = np.array([pair[1] for pair in pairs])
        history = self.model.fit(centers, contexts, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1)
        return history

    def evaluate(self, pairs):
        centers = np.array([pair[0] for pair in pairs])
        contexts = np.array([pair[1] for pair in pairs])
        loss = self.model.evaluate(centers, contexts, verbose=0)
        print(f"Evaluation MSE Loss: {loss:.4f}")
        return loss

# -----------------------------
# Load dataset, generate pairs and train
# -----------------------------
def load_dataset_and_pairs():
    pairs_all = []
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    processed = 0
    for filepath in files:
        y = load_audio_fixed(filepath)
        if y is not None:
            mfcc_seq = audio_to_mfcc_sequence(y)
            pairs = generate_skipgram_pairs(mfcc_seq)
            pairs_all.extend(pairs)
            processed += 1
            if processed % 20 == 0:
                print(f"Processed {processed} files and generated {len(pairs_all)} pairs")
    print(f"Total processed files: {processed}, total pairs: {len(pairs_all)}")
    return pairs_all

# -----------------------------
# Plot training loss
# -----------------------------
def plot_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Skip-Gram Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_7_skipgram_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("=== Experiment 7: Skip-Gram Model Adapted for Audio ===")
    pairs = load_dataset_and_pairs()
    if len(pairs) == 0:
        print("No data pairs generated. Check dataset path and files.")
        return

    trainer = SkipGramTrainer()
    trainer.build_model()
    history = trainer.train(pairs)
    trainer.evaluate(pairs)
    plot_history(history)

    with open('experiment_7_skipgram_results.txt', 'w') as f:
        f.write("=== Experiment 7: Skip-Gram Audio Embedding Results ===")
        f.write(f"Total training pairs: {len(pairs)}")
        f.write(f"Final Training Loss (MSE): {history.history['loss'][-1]:.4f}")
        f.write(f"Final Validation Loss (MSE): {history.history['val_loss'][-1]:.4f}")

if __name__ == '__main__':
    main()
