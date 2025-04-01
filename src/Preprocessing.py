
import librosa
import librosa.display
import numpy as np
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment


def preprocess_audio(file_path, output_path, target_sr=16000, duration=45.0):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sr)

    # Noise reduction
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)

    # Normalize volume
    reduced_noise = reduced_noise / np.max(np.abs(reduced_noise))

    # Trim silence
    trimmed_audio, _ = librosa.effects.trim(reduced_noise, top_db=20)

    # Ensure consistent duration (pad or truncate to fixed length)
    target_length = int(target_sr * duration)
    if len(trimmed_audio) < target_length:
        padded_audio = np.pad(trimmed_audio, (0, target_length - len(trimmed_audio)))
    else:
        padded_audio = trimmed_audio[:target_length]

    # Save processed file
    sf.write(output_path, padded_audio, samplerate=target_sr)
    print(f"Processed and saved: {output_path}")


# Example usage
input_file = r"C:\Users\Jeylani\Documents\capstone project\.venv\Audio\Unprocessed\SOM200_sentence1.wav"  # Change this to your file path
output_file = r"C:\Users\Jeylani\Documents\capstone project\.venv\Audio\Processed\SOM200_processed.wav"
preprocess_audio(input_file, output_file)
