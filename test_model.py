import tensorflow as tf
import numpy as np
from audio_utils import load_audio_fixed, wav_to_melspec, wav_to_mfcc_seq, pad_or_trim_time_axis

class_names = ["angry","disgust","fearful","happy","neutral","sad"]

def predict_file(wav_path, model, model_type="cnn"):
    y = load_audio_fixed(wav_path)

    # CNN branch
    mel = wav_to_melspec(y)
    mel = pad_or_trim_time_axis(mel.T, 188).T
    mel = np.expand_dims(mel, -1)[None, ...]

    # LSTM branch
    mfcc = wav_to_mfcc_seq(y)
    mfcc = pad_or_trim_time_axis(mfcc, 188)
    mfcc = mfcc[None, ...]

    if model_type == "cnn":
        probs = model.predict(mel, verbose=0)[0]
    elif model_type == "lstm":
        probs = model.predict(mfcc, verbose=0)[0]
    else:
        raise ValueError("model_type must be 'cnn' or 'lstm'")
    return dict(zip(class_names, probs.tolist()))

if __name__ == "__main__":
    model = tf.keras.models.load_model("./saved_models/best_cnn.keras")
    result = predict_file("C:\\Users\\arjun\\Downloads\\Music\\476646__ffmmendoza90__why.wav", model, model_type="cnn")
    print(result)
