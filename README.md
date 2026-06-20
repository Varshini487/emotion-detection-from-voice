# 🎤 Emotion Detection from Voice

Classify emotional states (anger, joy, sadness, fear, neutral) from audio using speech emotion recognition.

## 🧠 How It Works
1. Extract MFCC (Mel-Frequency Cepstral Coefficients) from audio — human-like frequency perception
2. Compute statistics: mean, std, min, max for each MFCC → 52-dimensional feature vector
3. Random Forest classifier predicts emotion
4. Output: emotion label + confidence score

## 📊 Performance
- Accuracy: 84%
- Emotions: Anger, Joy, Sadness, Fear, Neutral
- Audio format: WAV, MP3 (16-bit, 16kHz preferred)

## 🛠️ Tech Stack
- **librosa** – MFCC feature extraction
- **scikit-learn** – Random Forest classifier
- **TensorFlow** – LSTM variant for comparison
- **Streamlit** – web interface

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/emotion-detection-from-voice
cd emotion-detection-from-voice
pip install -r requirements.txt
streamlit run app.py
```

## 💡 Use Cases
- Call center emotion analysis (customer satisfaction)
- Mental health monitoring
- Accessibility tools (mood detection for assistive tech)
- Gaming (emotion-responsive NPCs)
