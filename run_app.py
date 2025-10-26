#!/usr/bin/env python3
"""
Audio Emotion Detection Web Application
Run this script to start the web server
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import tensorflow
        import librosa
        import numpy
        import sklearn
        import soundfile
        print("Core packages are installed")
        
        # Check speech recognition separately
        try:
            import speech_recognition
            print("Speech recognition available")
        except ImportError:
            print("Warning: Speech recognition not available (speech-to-text disabled)")
        
        return True
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the trained model exists"""
    model_path = "saved_models/best_cnn.keras"
    if os.path.exists(model_path):
        print("Model found")
        return True
    else:
        print(f"Model not found at {model_path}")
        print("Please train the model first by running: python aud_recognition.py")
        return False

def main():
    print("Audio Emotion Detection Web App")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model
    if not check_model():
        return
    
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Import and run the app
    try:
        from app import app, load_model
        load_model()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()