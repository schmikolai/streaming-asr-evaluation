import os
from src.eval.SampleResult import SampleResult
from typing import List


class Run:
    def __init__(self, samples: List[SampleResult]):
        self.samples = samples

    @classmethod
    def from_directory(cls, directory: str):
        samples = cls.__load_samples(directory)
        return cls(samples)

    @classmethod
    def __load_samples(cls, directory: str) -> List[SampleResult]:
        files = os.listdir(directory)
        files = [f for f in files if f.endswith("final.json")]

        # get ids from the first part of the filename separated by "_"
        file_ids = [f.split("_")[0] for f in files]

        samples = [SampleResult.load_by_id(directory, file_id) for file_id in file_ids]
        return samples
    
    def build_metrics(self, normalize_words: bool = True, align_to: str = "mfa", temporal_tolerance: float = 0.1):
        for sample in self.samples:
            sample.build_alignments(normalize_words=normalize_words, align_to=align_to, temporal_tolerance=temporal_tolerance)
            sample.word_error_rate()
        
        