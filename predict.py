import librosa
import numpy as np
from tensorflow.keras.models import load_model
import argparse

# Feature extraction function
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# Predict function
def predict(file_path, model_path='models/digit_recognition_model.h5'):
    # Load model
    model = load_model(model_path)
    # Extract features
    features = extract_features(file_path)
    # Predict
    prediction = model.predict(features)
    digit = np.argmax(prediction)
    print(f"Predicted Digit: {digit}")

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict spoken digit from audio file")
    parser.add_argument('--file', required=True, help="Path to the audio file")
    args = parser.parse_args()
    predict(args.file)
