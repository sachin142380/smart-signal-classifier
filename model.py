from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import generate_signal, extract_features
from sklearn.metrics import confusion_matrix

def train_model():
    X, y = [], []
    
    signal_types = ["sine", "square", "triangle", "sawtooth",
    "chirp", "noise", "pulse", "damped",
    "am", "fm", "gaussian", "spike",
    "step", "random_walk", "burst"]
    
    for _ in range(100):
        for s in signal_types:
            sig = generate_signal(s)
            X.append(extract_features(sig))
            y.append(s)
    
    X = np.array(X)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    
    return model, scaler, X_scaled, y   # ✅ RETURN THIS


def get_conf_matrix(model, X, y):
    y_pred = model.predict(X)
    return confusion_matrix(y, y_pred)