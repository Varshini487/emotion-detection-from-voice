# 🎤 Emotion Detection from Voice

Detect emotional states (anger, joy, sadness, fear, neutral) from speech audio using machine learning and audio feature extraction.

## How it Works

1. **Audio Feature Extraction** (MFCC)
   - Mel-Frequency Cepstral Coefficients capture voice characteristics
   - 13 MFCCs × time steps = 2D feature matrix

2. **Temporal Modeling**
   - Random Forest on aggregated MFCC statistics (mean, std, min, max)
   - OR LSTM for sequence modeling (captures emotion dynamics)

3. **Classification**
   - 5-class emotion: Anger, Joy, Sadness, Fear, Neutral
   - Output: emotion label + confidence (0-100%)

## Dataset
- RAVDESS (Ryerson Audio-Visual Emotion Speech): 1,440 utterances × 8 emotions
- Speech Emotion Recognition 2019 (SER): 13,000 audio samples

## Tech Stack
- **librosa** – MFCC extraction
- **scikit-learn** – Random Forest, SVM
- **TensorFlow** – LSTM variant
- **Streamlit** – real-time emotion demo

## Performance
- Random Forest: 78% accuracy
- LSTM: 84% accuracy (captures temporal patterns in voice)

## Key Insights
- Pitch & energy are strong anger indicators
- Slower speech + lower energy = sadness
- MFCC aggregation reduces 10,000 features to 13—still works!

## Use Cases
- Call center customer satisfaction monitoring
- Mental health screening tools
- Voice-controlled AI with emotion awareness
- Music generation conditioned on listener emotion
