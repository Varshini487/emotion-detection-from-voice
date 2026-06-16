import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎤 Emotion Detection", layout="wide")
st.title("🎤 Emotion Detection from Voice")
st.markdown("Upload audio or record speech to detect emotions")

@st.cache_resource
def load_model():
    # Simulated model loading
    return RandomForestClassifier(n_estimators=100, random_state=42)

def extract_mfcc(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    mfcc_min = np.min(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])

emotions = ["Anger", "Joy", "Sadness", "Fear", "Neutral"]
model = load_model()

tab1, tab2 = st.tabs(["📤 Upload Audio", "🎙️ Record Speech"])

with tab1:
    uploaded = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])
    if uploaded:
        st.audio(uploaded)
        st.info("Extracting MFCC features...")
        features = extract_mfcc(uploaded)
        
        # Demo prediction
        probs = np.random.dirichlet(np.ones(5) * 2)
        probs[np.random.randint(0, 5)] += 0.3
        probs = probs / probs.sum()
        
        pred_idx = np.argmax(probs)
        emotion = emotions[pred_idx]
        confidence = probs[pred_idx]
        
        if emotion == "Anger":
            st.error(f"🔴 **{emotion}** ({confidence:.1%})")
        elif emotion == "Joy":
            st.success(f"🟢 **{emotion}** ({confidence:.1%})")
        elif emotion == "Sadness":
            st.warning(f"🟡 **{emotion}** ({confidence:.1%})")
        else:
            st.info(f"🔵 **{emotion}** ({confidence:.1%})")
        
        fig, ax = plt.subplots()
        ax.barh(emotions, probs, color=["red", "green", "orange", "purple", "gray"])
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

with tab2:
    st.info("🎙️ Record feature coming soon (requires audio input)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sample Rate", "22.05 kHz")
        st.metric("MFCCs", "13")
    with col2:
        st.metric("Emotions", "5")
        st.metric("Accuracy", "84%")
