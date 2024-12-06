import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract MFCC features from an audio file.
    """
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
    return np.mean(mfcc, axis=1)
