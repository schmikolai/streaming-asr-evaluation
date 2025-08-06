import os
from src.eval.SampleResult import SampleResult
from typing import List
from tqdm import tqdm


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
        for sample in tqdm(self.samples, desc="Building metrics"):
            sample.build_alignments(normalize_words=normalize_words, align_to=align_to, temporal_tolerance=temporal_tolerance)
            sample.word_error_rate()
            sample.word_first_corrects()
            sample.word_first_finals()
        
        self.wer = sum(sample.word_error_rate() for sample in self.samples) / len(self.samples)
        
        self.wfc_latency = sum(
            sum(
                wfc["latency"] for wfc in sample.wfc if wfc is not None
            ) for sample in self.samples
        ) / sum(
            len(sample.wfc) for sample in self.samples
        )

        self.wff_latency = sum(
            sum(
                wff["latency"] for wff in sample.wff if wff is not None
            ) for sample in self.samples
        ) / sum(
            len(sample.wff) for sample in self.samples
        )
