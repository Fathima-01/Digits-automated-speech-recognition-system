import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset path
DATASET_PATH = 'free-spoken-digit-dataset/recordings'

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
    return np.mean(mfcc, axis=1)

# Load dataset
def load_data():
    files = os.listdir(DATASET_PATH)
    features, labels = [], []
    for file in files:
        if file.endswith('.wav'):
            feature = extract_features(os.path.join(DATASET_PATH, file))
            features.append(feature)
            label = int(file.split('_')[0])  # Extract digit from filename
            labels.append(label)
    return np.array(features), np.array(labels)

# Load data
X, y = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('models/digit_recognition_model.h5')
