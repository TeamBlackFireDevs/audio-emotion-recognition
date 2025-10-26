import librosa, numpy as np

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

def load_audio_fixed(path, target_sr=TARGET_SR, target_len=SAMPLES_PER_CLIP):
    y, sr = librosa.load(path, sr=target_sr)
    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    else:
        y = y[:target_len]
    return y

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
