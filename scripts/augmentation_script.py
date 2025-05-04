import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

input_dir = r"C:\Users\Jeylani\Documents\capstone project\.venv\Audio\NonSomali-Processed"
output_dir = r"C:\Users\Jeylani\Documents\capstone project\.venv\Audio\nonSomaliAugmented"
os.makedirs(output_dir, exist_ok=True)

def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise

def change_pitch(data, sr, pitch_factor):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def change_speed(data, speed_factor):
    return librosa.effects.time_stretch(data, rate=speed_factor)

for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".wav"):
        path = os.path.join(input_dir, filename)
        data, sr = librosa.load(path, sr=None)

        # Save original as-is
        sf.write(os.path.join(output_dir, filename), data, sr)

        # Augmentation 1: Add noise
        noisy = add_noise(data)
        sf.write(os.path.join(output_dir, f"{filename[:-4]}_noise.wav"), noisy, sr)

        # Augmentation 2: Pitch up
        pitch_up = change_pitch(data, sr, pitch_factor=2)
        sf.write(os.path.join(output_dir, f"{filename[:-4]}_pitchup.wav"), pitch_up, sr)

        # Augmentation 3: Pitch down
        pitch_down = change_pitch(data, sr, pitch_factor=-2)
        sf.write(os.path.join(output_dir, f"{filename[:-4]}_pitchdown.wav"), pitch_down, sr)

        # Augmentation 4: Speed up
        if len(data) > 2:  # Avoid short clips crash
            fast = change_speed(data, 1.2)
            sf.write(os.path.join(output_dir, f"{filename[:-4]}_fast.wav"), fast, sr)

        # Augmentation 5: Slow down
        if len(data) > 2:
            slow = change_speed(data, 0.8)
            sf.write(os.path.join(output_dir, f"{filename[:-4]}_slow.wav"), slow, sr)
