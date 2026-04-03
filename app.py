import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.fft import fft
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import train_model
from utils import generate_signal, extract_features, t
from cnn_model import train_cnn

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Signal Analyzer", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>📊 AI Signal Analyzer</h1>
<p style='text-align:center; color:gray;'>ML + Deep Learning Dashboard</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

signal_types = [
    "sine", "square", "triangle", "sawtooth",
    "chirp", "noise", "pulse", "damped",
    "am", "fm", "gaussian", "spike",
    "step", "random_walk", "burst"
]

signal_type = st.sidebar.selectbox("Select Signal", signal_types)
freq = st.sidebar.slider("Frequency", 1, 10, 5)
noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)

uploaded_file = st.sidebar.file_uploader("Upload Audio (.wav)", type=["wav"])

# 🔥 CNN TOGGLE
use_cnn = st.sidebar.checkbox("Use Deep Learning (CNN)")

# ---------------- LOAD MODEL ----------------
if use_cnn:
    model = train_cnn(generate_signal, signal_types)
    scaler = None
    X, y = None, None
else:
    model, scaler, X, y = train_model()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Signal", "🎧 Audio", "📈 Analytics"])

# ================= SIGNAL TAB =================
with tab1:
    
    st.subheader("📊 Signal Analysis")
    
    if st.button("Predict Signal"):
        
        sig = generate_signal(signal_type, freq)
        sig = sig + np.random.normal(0, noise, len(sig))
        
        # 🔥 CNN vs ML switch
        if use_cnn:
            input_sig = sig.reshape(1, 500, 1)
            pred_idx = np.argmax(model.predict(input_sig), axis=1)[0]
            pred = signal_types[pred_idx]
        else:
            features = extract_features(sig)
            features = scaler.transform([features])
            pred = model.predict(features)[0]
        
        st.success(f"Predicted Signal: {pred}")
        
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.plot(t, sig)
            ax.set_title("Time Domain")
            st.pyplot(fig)

        with col2:
            fft_vals = np.abs(fft(sig))
            fig2, ax2 = plt.subplots()
            ax2.plot(fft_vals)
            ax2.set_title("FFT Spectrum")
            st.pyplot(fig2)

# ================= AUDIO TAB =================
with tab2:
    
    st.subheader("🎧 Audio Analysis")
    
    if uploaded_file is not None:
        
        audio, sr = sf.read(uploaded_file)
        
        st.write(f"Sample Rate: {sr}")
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        audio = audio[:500]
        
        fig, ax = plt.subplots()
        ax.plot(audio)
        ax.set_title("Audio Signal")
        st.pyplot(fig)
        
        # 🔥 CNN vs ML switch
        if use_cnn:
            input_sig = audio.reshape(1, 500, 1)
            pred_idx = np.argmax(model.predict(input_sig), axis=1)[0]
            pred = signal_types[pred_idx]
        else:
            features = extract_features(audio)
            features = scaler.transform([features])
            pred = model.predict(features)[0]
        
        st.success(f"Predicted Audio Type: {pred}")

# ================= ANALYTICS TAB =================
with tab3:
    
    st.subheader("📈 Model Analytics")
    
    if not use_cnn:
        if st.button("Show Confusion Matrix"):
            
            y_pred = model.predict(X)
            cm = confusion_matrix(y, y_pred)
            
            fig, ax = plt.subplots(figsize=(8,6))
            
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues"
                xticklabels=np.unique(y),
                yticklabels=np.unique(y)
            )
            
            st.pyplot(fig)
    else:
        st.info("Confusion matrix disabled for CNN (can add later)")
        
# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ by Sachin 🚀")