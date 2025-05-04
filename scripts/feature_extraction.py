import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


# Define feature extraction function
def extract_features(file_path, sr=16000, n_mfcc=13):
    try:
        print(f"Loading: {file_path}")  # Debugging line
        audio, sr = librosa.load(file_path, sr=sr)

        if len(audio) == 0:
            print(f"Warning: Empty audio file {file_path}")
            return None

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Extract spectral features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

        # Extract ZCR & RMSE
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rmse = np.mean(librosa.feature.rms(y=audio))

        # Combine features into a single list
        features = np.hstack([mfccs_mean, chroma, spec_centroid, spec_bandwidth, zcr, rmse])
        print(f"Extracted features: {features.shape}")  # Debugging line
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Set input directory where the Somali voice recordings are stored
input_dir = "./Audio/Processed"
output_csv = "extracted_features.csv"

# Check if input directory exists
if not os.path.exists(input_dir):
    print(f"Error: Directory {input_dir} does not exist.")
    exit()

# Process all audio files
feature_list = []
file_names = []

print("Extracting features from audio files...")
for file_name in tqdm(os.listdir(input_dir)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(input_dir, file_name)
        print(f"Processing: {file_name}")  # Debugging line
        features = extract_features(file_path)
        if features is not None:
            feature_list.append(features)
            file_names.append(file_name)

# Convert to DataFrame
columns = [f"MFCC_{i + 1}" for i in range(13)] + ["Chroma", "Spectral_Centroid", "Spectral_Bandwidth", "ZCR", "RMSE"]
df = pd.DataFrame(feature_list, columns=columns)
df.insert(0, "Filename", file_names)  # Add filename column

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Feature extraction completed! Saved to {output_csv}")