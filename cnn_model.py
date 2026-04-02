import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

def build_cnn(input_shape, num_classes):
    
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
def train_cnn(generate_signal, signal_types):
    
    X = []
    y = []
    
    for i, sig_type in enumerate(signal_types):
        for _ in range(100):
            sig = generate_signal(sig_type)
            X.append(sig)
            y.append(i)
    
    X = np.array(X)
    X = X.reshape(-1, 500, 1)   # CNN input
    
    y = np.array(y)
    
    model = build_cnn((500,1), len(signal_types))
    
    model.fit(X, y, epochs=5, verbose=0)
    
    return model