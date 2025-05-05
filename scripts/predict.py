import librosa
import numpy as np
import joblib
import sys
import os
import pandas as pd

# Load model and scaler
try:
    model_path = "models/svm_model.joblib"
    scaler_path = "models/scaler.joblib"

    print(f"Loading model from: {model_path}")
    print(f"Loading scaler from: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    sys.exit(1)


def extract_features(audio_path):
    """Extract audio features exactly matching the 18 features expected by the model"""
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features matching the model's expected feature names
    # MFCC_1 through MFCC_13
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Single chroma feature (aggregated)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(np.mean(chroma, axis=1))  # Single chroma value

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # RMS energy
    rms = librosa.feature.rms(y=y)

    # Combine features in the exact order matching the model
    feature_values = np.hstack([
        np.mean(mfccs, axis=1),  # 13 MFCC features
        chroma_mean,  # 1 Chroma feature
        np.mean(spec_centroid),  # 1 Spectral Centroid feature
        np.mean(spec_bandwidth),  # 1 Spectral Bandwidth feature
        np.mean(zcr),  # 1 ZCR feature
        np.mean(rms)  # 1 RMSE feature
    ])

    # Create feature names that match training data
    feature_names = []
    for i in range(13):
        feature_names.append(f'MFCC_{i + 1}')
    feature_names.extend(['Chroma', 'Spectral_Centroid', 'Spectral_Bandwidth', 'ZCR', 'RMSE'])

    # Create pandas DataFrame with named features (matching training data)
    features_df = pd.DataFrame([feature_values], columns=feature_names)

    return features_df


def predict(audio_path):
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return

    print(f"\nAnalyzing audio file: {audio_path}")
    features = extract_features(audio_path)

    # Verify we have the correct number of features
    print(f"Extracted {features.shape[1]} features (model expects 18)")

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    # Get decision function value (distance from hyperplane) for confidence approximation
    if hasattr(model, 'decision_function'):
        decision_value = model.decision_function(scaled)[0]
        # Convert decision value to a confidence-like score (0-1 range)
        confidence = 1 / (1 + np.exp(-abs(decision_value)))
    else:
        confidence = None

    label = "Somali" if prediction == 1 else "Non-Somali"

    print("\n====== PREDICTION RESULT ======")
    print(f"Speech classified as: {label}")
    if confidence is not None:
        print(f"Confidence score: {confidence:.2%}")
    print("===============================")


# Example usage
if __name__ == "__main__":
    # Check if a file path was provided as a command line argument
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        # If no command line argument, use the hardcoded path
        audio_path = r"C:\Users\Jeylani\Documents\capstone project\.venv\Audio\NonSomali-Processed\NonSOM010_processed.wav"
        print(f"No path provided as argument, using default path: {audio_path}")
        predict(audio_path)