from src.run.Dataset import Dataset
from matplotlib import pyplot as plt
import numpy as np
import os

def draw_wave(sample_id,
         start_time,
         end_time,
         dataset_dir="../data",
         dataset="librispeech-pc-test-clean",
         sample_rate=16000,
         bytes_per_sample=2,
         ax: plt.Axes = None
        ):

    audio_path = os.path.join(dataset_dir, dataset, sample_id, sample_id + ".mp3")

    # Load audio file
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file {audio_path} does not exist.")
    
    audio_bytes = Dataset.mp3_to_waveform(audio_path, sample_rate)

    start_index = int(start_time * sample_rate * bytes_per_sample)
    end_index = int(((end_time * sample_rate * bytes_per_sample) // bytes_per_sample) * bytes_per_sample)
    audio_bytes = audio_bytes[start_index:end_index]

    audio_samples = np.frombuffer(audio_bytes, dtype=np.int16)

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 3))
        ax.set_xlim(start_time, end_time)
        ax.xaxis.tick_top()
    ax.plot(np.linspace(start_time, end_time, len(audio_samples)), audio_samples, color='blue', linewidth=0.5)