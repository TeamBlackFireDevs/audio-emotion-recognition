class AudioEmotionDetector {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.emotionIcons = {
            'angry': 'fas fa-angry',
            'disgust': 'fas fa-grimace',
            'fearful': 'fas fa-frown-open',
            'happy': 'fas fa-smile-beam',
            'neutral': 'fas fa-meh',
            'sad': 'fas fa-sad-tear'
        };
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.audioFile = document.getElementById('audioFile');
        this.filePreview = document.getElementById('filePreview');
        this.fileName = document.getElementById('fileName');
        this.audioPlayer = document.getElementById('audioPlayer');
        this.removeFile = document.getElementById('removeFile');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.loading = document.getElementById('loading');
        this.transcriptSection = document.getElementById('transcriptSection');
        this.transcriptText = document.getElementById('transcriptText');
        this.resultsSection = document.getElementById('resultsSection');
        this.emotionGrid = document.getElementById('emotionGrid');
        this.newAnalysisBtn = document.getElementById('newAnalysisBtn');
        this.errorMessage = document.getElementById('errorMessage');
        this.errorText = document.getElementById('errorText');
    }

    bindEvents() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.audioFile.click());
        this.audioFile.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Button events
        this.removeFile.addEventListener('click', () => this.clearFile());
        this.analyzeBtn.addEventListener('click', () => this.analyzeAudio());
        this.newAnalysisBtn.addEventListener('click', () => this.resetForNewAnalysis());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac', 'audio/x-m4a'];
        const validExtensions = ['.wav', '.mp3', '.m4a', '.flac'];
        
        const isValidType = validTypes.includes(file.type) || 
                           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!isValidType) {
            this.showError('Please select a valid audio file (WAV, MP3, M4A, or FLAC)');
            return;
        }

        // Validate file size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            return;
        }

        this.displayFilePreview(file);
    }

    displayFilePreview(file) {
        this.fileName.textContent = file.name;
        
        // Create audio URL for preview
        const audioURL = URL.createObjectURL(file);
        this.audioPlayer.src = audioURL;
        
        // Show preview and hide upload area
        this.uploadArea.style.display = 'none';
        this.filePreview.style.display = 'block';
        this.analyzeBtn.disabled = false;
        
        // Store file for later use
        this.selectedFile = file;
        
        this.hideError();
    }

    clearFile() {
        this.uploadArea.style.display = 'block';
        this.filePreview.style.display = 'none';
        this.analyzeBtn.disabled = true;
        this.audioFile.value = '';
        this.selectedFile = null;
        
        // Revoke object URL to free memory
        if (this.audioPlayer.src) {
            URL.revokeObjectURL(this.audioPlayer.src);
        }
    }

    async analyzeAudio() {
        if (!this.selectedFile) {
            this.showError('Please select an audio file first');
            return;
        }

        this.showLoading();
        this.hideError();

        try {
            const formData = new FormData();
            formData.append('audio', this.selectedFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.displayTranscript(data.transcript);
                this.displayResults(data.predictions);
            } else {
                this.showError(data.error || 'Failed to analyze audio');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(predictions) {
        // Sort emotions by percentage (descending)
        const sortedEmotions = Object.entries(predictions)
            .sort(([,a], [,b]) => b - a);

        this.emotionGrid.innerHTML = '';

        sortedEmotions.forEach(([emotion, percentage], index) => {
            const card = this.createEmotionCard(emotion, percentage);
            this.emotionGrid.appendChild(card);
            
            // Animate cards with delay
            setTimeout(() => {
                card.classList.add('animate');
                const progressFill = card.querySelector('.progress-fill');
                progressFill.style.width = `${percentage}%`;
            }, index * 150);
        });

        this.resultsSection.style.display = 'block';
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    createEmotionCard(emotion, percentage) {
        const card = document.createElement('div');
        card.className = 'emotion-card';
        
        const iconClass = this.emotionIcons[emotion] || 'fas fa-question';
        const emotionClass = `emotion-${emotion}`;
        
        card.innerHTML = `
            <div class="emotion-icon ${emotionClass}">
                <i class="${iconClass}"></i>
            </div>
            <div class="emotion-name">${emotion}</div>
            <div class="emotion-percentage">${percentage.toFixed(1)}%</div>
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
        `;
        
        return card;
    }

    displayTranscript(transcript) {
        this.transcriptText.textContent = transcript || 'No speech detected';
        this.transcriptSection.style.display = 'block';
        
        // Scroll to transcript
        this.transcriptSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    showLoading() {
        this.loading.style.display = 'block';
        this.transcriptSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
    }

    hideLoading() {
        this.loading.style.display = 'none';
    }

    showError(message) {
        this.errorText.textContent = message;
        this.errorMessage.style.display = 'block';
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        this.errorMessage.style.display = 'none';
    }

    resetForNewAnalysis() {
        this.clearFile();
        this.transcriptSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.hideError();
        
        // Scroll back to top
        window.scrollTo({ 
            top: 0, 
            behavior: 'smooth' 
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AudioEmotionDetector();
});

// Add some visual feedback for better UX
document.addEventListener('DOMContentLoaded', () => {
    // Add loading animation to buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = '';
                }, 150);
            }
        });
    });

    // Add smooth transitions to all interactive elements
    const interactiveElements = document.querySelectorAll('button, .upload-area, .emotion-card');
    interactiveElements.forEach(element => {
        element.style.transition = 'all 0.3s ease';
    });
});