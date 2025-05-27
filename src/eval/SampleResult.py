from dataclasses import dataclass
import os
import json
from textgrid import TextGrid, Interval
from typing import Literal

from src.eval.PredictionAlignment import PredictionAlignment
from src.eval.metrics.word_first_correct import word_first_correct_response
from src.eval.metrics.word_first_final import word_first_final_response

@dataclass
class WordResult:
    word: str
    conf: float
    start: float
    end: float


@dataclass
class FinalMessage:
    result: list[WordResult]
    reason: str
    observation_time: float


@dataclass
class PartialResult:
    result: list[WordResult]
    window: tuple[float, float]
    observation_time: float


class SampleResult:
    sample_id: str
    final: list[WordResult]
    final_messages: list[FinalMessage]
    partials: list[PartialResult]
    baseline: list[WordResult] = None
    alignments: list[PredictionAlignment] = None
    transcript: str = None
    mfa: list[WordResult] = None

    _alignment_sequence: list[WordResult] = None

    def __init__(
        self,
        sample_id: str,
        final: list[WordResult],
        final_messages: list[FinalMessage],
        partials: list[PartialResult],
        baseline: list[WordResult] = None,
        transcript: list[WordResult] = None,
        mfa: list[WordResult] = None,
    ):
        self.sample_id = sample_id
        self.final = final
        self.final_messages = final_messages
        self.partials = partials
        self.baseline = baseline
        self.transcript = transcript
        self.mfa = mfa

    @classmethod
    def load_by_id(cls, directory: str, sample_id: str):
        def parse_word_result_list(data):
            return [WordResult(**w) for w in data]

        def parse_final_messages(data):
            return [
                FinalMessage(
                    result=parse_word_result_list(msg["result"]),
                    reason=msg["reason"],
                    observation_time=msg["observation_time"],
                )
                for msg in data
            ]

        def parse_partials(data):
            return [
                PartialResult(
                    result=[WordResult(**w) for w in p["result"]["result"]],
                    window=tuple(p["window"]),
                    observation_time=p["observation_time"],
                )
                for p in data
            ]
        
        def parse_transcript(string: str):
            words = string.split()
            return [WordResult(word=w, conf=1.0, start=0.0, end=0.0) for w in words]
        
        def parse_mfa(data: list[Interval]):
            return [
                WordResult(
                    word=w.mark,
                    conf=1.0,
                    start=w.minTime,
                    end=w.maxTime,
                )
                for w in data
                if w.mark != ""
            ]

        with open(os.path.join(directory, sample_id + "_final.json"), "r") as f:
            final_data = json.load(f)

        with open(
            os.path.join(directory, sample_id + "_final_messages.json"), "r"
        ) as f:
            final_messages_data = json.load(f)

        with open(os.path.join(directory, sample_id + "_partial.json"), "r") as f:
            partials_data = json.load(f)

        with open(os.path.join("../out/baseline", sample_id + ".json"), "r") as f:
            baseline_data = json.load(f)
        
        with open(os.path.join("../data/librispeech-pc-test-clean", sample_id, sample_id + ".txt"), "r") as f:
            transcript_str = f.read().strip()
        
        tg = TextGrid.fromFile(os.path.join("../data/mfa", sample_id, sample_id + ".TextGrid"), "r")
        mfa_data = tg.getFirst("words").intervals

        obj = cls(
            sample_id=sample_id,
            final=parse_word_result_list(final_data),
            final_messages=parse_final_messages(final_messages_data),
            partials=parse_partials(partials_data),
            baseline=parse_word_result_list(baseline_data),
            transcript=parse_transcript(transcript_str),
            mfa=parse_mfa(mfa_data),
        )

        return obj

    def build_alignments(self,
                         normalize_words=True,
                         align_to: Literal["final", "baseline", "mfa"]="final",
                         temporal_tolerance: float=0.5):
        if align_to == "baseline":
            if self.baseline is None:
                raise ValueError("Baseline is not set in the sample.")
            self._alignment_sequence = self.baseline
        elif align_to == "mfa":
            if self.mfa is None:
                raise ValueError("MFA is not set in the sample.")
            self._alignment_sequence = self.mfa
        elif align_to == "final":
            self._alignment_sequence = self.final
        else:
            raise ValueError("Invalid value for align_to. Use 'final', 'mfa' or 'baseline'.")
        
        self.alignments = []
        for timestep in range(len(self.partials)):
            alignment = PredictionAlignment(self,
                                            timestep,
                                            normalize_words=normalize_words,
                                            temporal_tolerance=temporal_tolerance)
            alignment.build()
            self.alignments.append(alignment)
        return self
    
    def word_first_corrects(self):
        return [
            word_first_correct_response(self._alignment_sequence, self.partials, i, self.alignments)
            for i in range(len(self._alignment_sequence))
        ]
    
    def word_first_finals(self):
        return [
            word_first_final_response(self._alignment_sequence, self.partials, i, self.alignments)
            for i in range(len(self._alignment_sequence))
        ]
