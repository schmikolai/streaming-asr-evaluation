from typing import Tuple
from os import path, listdir
import subprocess
import logging
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, dataset_name: str = "librispeech-pc-test-clean", datasets_dir: str = "data"):
        """
        Initializes the Dataset with the given dataset name and directory.
        Args:
            dataset_name (str): The name of the dataset to load. Defaults to "librispeech-pc-test-clean". Possible values are "librispeech-pc-test-clean" and "librispeech-pc-test-other".
            datasets_dir (str): The directory where the dataset is located. Defaults to "data".
        """
        self.dataset_path = path.join(datasets_dir, dataset_name)
        entries = listdir(self.dataset_path)
        entries = [entry for entry in entries if path.isdir(path.join(self.dataset_path, entry))]
        self.entries_iter = iter(entries)

        logger.info(f"Loaded dataset {dataset_name} with {len(entries)} elements")

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[str, bytes, str]:
        element_id = next(self.entries_iter)
        element_path = path.join(self.dataset_path, element_id)

        logger.debug(f"Loading dataset element with ID {element_id}")

        audio_file = path.join(element_path, f"{element_id}.mp3")
        transcript_file = path.join(element_path, f"{element_id}.txt")

        audio_bytes = Dataset.mp3_to_waveform(audio_file)

        with open(transcript_file, "r") as f:
            transcript = f.read()
            return element_id, audio_bytes, transcript

    @staticmethod
    def mp3_to_waveform(mp3_file, sample_rate=16000) -> bytes:
        # FFmpeg-command to read mp3 as bytes
        process = subprocess.Popen(
            ["ffmpeg", "-i", mp3_file, "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        raw_data = process.stdout.read()
        logger.debug(f"Read {len(raw_data)} bytes from {mp3_file}")

        return raw_data