import librosa
import numpy as np
import joblib
import sys
import os

# Load model and scaler
model = joblib.load("models/svm_model.joblib")
scaler = joblib.load("models/scaler.joblib")
audio_path = r"C:\Users\Jeylani\Documents\capstone project\.venv\Audio\NonSomali-Processed\NonSOM001_processed.wav"


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Feature extraction (same logic as used during training)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Aggregate (mean) to form a single feature vector
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr),
        np.mean(rms),
        np.mean(spec_centroid)
    ])
    return features.reshape(1, -1)

def predict(audio_path):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    features = extract_features(audio_path)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    label = "Somali" if prediction == 1 else "non-Somali"

    print(f"Prediction: {label} (class {prediction})")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_audio.wav")
    else:
        predict(sys.argv[1])
