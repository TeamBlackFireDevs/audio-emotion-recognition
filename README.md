# 🎵 Audio Emotion Detection Web Application

A modern, responsive web application for detecting emotions in audio files using deep learning. Built with Flask backend and vanilla JavaScript frontend.

## ✨ Features

- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Drag & Drop Upload**: Easy file upload with drag-and-drop support
- **Real-time Preview**: Audio player to preview uploaded files
- **Beautiful Visualizations**: Animated emotion percentage charts
- **Multiple Formats**: Supports WAV, MP3, M4A, and FLAC files
- **Fast Processing**: Optimized model inference for quick results

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already done)
```bash
python aud_recognition.py
```

### 3. Start the Web Application
```bash
python run_app.py
```

### 4. Open Your Browser
Navigate to: `http://localhost:5000`

## 📁 Project Structure

```
Audio Emotion Detection/
├── app.py                 # Flask web application
├── run_app.py            # Application launcher
├── aud_recognition.py    # Model training script
├── audio_utils.py        # Audio processing utilities
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Main web interface
├── static/
│   ├── style.css        # Responsive CSS styles
│   └── script.js        # Interactive JavaScript
└── saved_models/
    └── best_cnn.keras   # Trained emotion detection model
```

## 🎯 Supported Emotions

The model can detect 6 different emotions:
- 😠 **Angry**
- 🤢 **Disgust** 
- 😨 **Fearful**
- 😊 **Happy**
- 😐 **Neutral**
- 😢 **Sad**

## 🔧 Technical Details

### Backend (Flask)
- **Framework**: Flask 2.3.3
- **ML Library**: TensorFlow 2.13.0
- **Audio Processing**: librosa 0.10.1
- **Model**: CNN trained on mel-spectrograms

### Frontend
- **Styling**: Pure CSS with modern gradients and animations
- **JavaScript**: Vanilla JS with ES6+ features
- **Icons**: Font Awesome 6.0
- **Fonts**: Google Fonts (Poppins)

### Audio Processing Pipeline
1. **Load Audio**: Normalize to 16kHz, 3-second clips
2. **Feature Extraction**: Generate mel-spectrograms (128 mel bands)
3. **Preprocessing**: Standardize features
4. **Prediction**: CNN model inference
5. **Results**: Emotion probabilities as percentages

## 📱 Responsive Design

The application is fully responsive and optimized for:
- **Desktop**: Full-featured interface with grid layout
- **Tablet**: Adapted layout with touch-friendly controls
- **Mobile**: Single-column layout with optimized interactions

## 🎨 UI/UX Features

- **Modern Gradient Background**: Eye-catching purple gradient
- **Glass Morphism**: Frosted glass effect on cards
- **Smooth Animations**: CSS transitions and keyframe animations
- **Interactive Feedback**: Hover effects and button animations
- **Progress Indicators**: Animated progress bars for results
- **Error Handling**: User-friendly error messages

## 🔒 Security Features

- **File Validation**: Strict file type and size checking
- **Secure Uploads**: Werkzeug secure filename handling
- **Memory Management**: Automatic cleanup of uploaded files
- **Input Sanitization**: Protection against malicious uploads

## 🚀 Performance Optimizations

- **Lazy Loading**: Model loaded only when needed
- **Efficient Processing**: Optimized audio feature extraction
- **Memory Management**: Automatic cleanup of temporary files
- **Caching**: Browser caching for static assets

## 🛠️ Development

### Running in Development Mode
```bash
python app.py
```

### Customizing the Model
To use a different model, update the `MODEL_PATH` in `app.py`:
```python
MODEL_PATH = 'path/to/your/model.keras'
```

### Adding New Emotions
1. Update `class_names` in `app.py`
2. Add corresponding icons in `script.js`
3. Add emotion-specific CSS classes in `style.css`

## 📊 Model Performance

The CNN model achieves:
- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Inference Time**: <1 second per file

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🆘 Troubleshooting

### Common Issues

**Model not found error:**
```bash
python aud_recognition.py  # Train the model first
```

**Package import errors:**
```bash
pip install -r requirements.txt  # Install dependencies
```

**Audio file not supported:**
- Ensure file is WAV, MP3, M4A, or FLAC format
- Check file size is under 16MB

**Server won't start:**
- Check if port 5000 is available
- Try running with different port: `app.run(port=8000)`

## 📞 Support

For issues or questions, please check the troubleshooting section above or create an issue in the repository.

---

Made with ❤️ for Audio Emotion Detection