from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
import librosa
from werkzeug.utils import secure_filename
from audio_utils import load_audio_fixed, wav_to_melspec, pad_or_trim_time_axis

# Try to import speech recognition, fallback if not available
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: SpeechRecognition not available. Speech-to-text feature disabled.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'saved_models/best_cnn.keras'
model = None
class_names = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {MODEL_PATH}")

def speech_to_text(audio_path):
    """Convert speech to text using speech recognition"""
    if not SPEECH_RECOGNITION_AVAILABLE:
        return "Speech recognition not available"
    
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Speech recognition service unavailable"
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return "Error processing speech"

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    try:
        # Load and preprocess audio
        y = load_audio_fixed(audio_path)
        
        # Generate mel-spectrogram
        mel = wav_to_melspec(y)
        mel = pad_or_trim_time_axis(mel.T, 188).T  # fix time axis to 188
        mel = np.expand_dims(mel, -1)[None, ...]   # (1, 128, 188, 1)
        
        # Predict
        probs = model.predict(mel, verbose=0)[0]
        
        # Create result dictionary
        results = {}
        for i, emotion in enumerate(class_names):
            results[emotion] = float(probs[i] * 100)  # Convert to percentage
            
        return results
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert speech to text
        transcript = speech_to_text(filepath)
        
        # Predict emotion
        results = predict_emotion(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        if results:
            return jsonify({
                'success': True, 
                'predictions': results,
                'transcript': transcript
            })
        else:
            return jsonify({'error': 'Failed to process audio file'}), 500
    else:
        return jsonify({'error': 'Invalid file format. Please upload WAV, MP3, M4A, or FLAC files.'}), 400

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)