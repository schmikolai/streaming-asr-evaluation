from dataclasses import dataclass
import os
import json
from textgrid import TextGrid, Interval
from typing import Literal
from joblib import Parallel, delayed
import warnings

import openwer

from src.eval.PredictionAlignment import PredictionAlignment
from src.eval.metrics.word_first_correct import word_first_correct_response
from src.eval.metrics.word_first_final import word_first_final_response
from src.helper.word_sequence import word_sequence_to_string

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
    final_mfa: list[WordResult] = None

    _alignment_sequence: list[WordResult] = None
    _wer = None
    wfc = None
    wff = None

    def __init__(
        self,
        sample_id: str,
        final: list[WordResult],
        final_messages: list[FinalMessage],
        partials: list[PartialResult],
        baseline: list[WordResult] = None,
        transcript: list[WordResult] = None,
        mfa: list[WordResult] = None,
        final_mfa: list[WordResult] = None,
    ):
        self.sample_id = sample_id
        self.final = final
        self.final_messages = final_messages
        self.partials = partials
        self.baseline = baseline
        self.transcript = transcript
        self.mfa = mfa
        self.final_mfa = final_mfa

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
            # Baseline
            baseline_path = os.path.join("../out/baseline", sample_id + ".json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                    baseline = parse_word_result_list(baseline_data)
            else:
                warnings.warn(f"Baseline file not found: {baseline_path}")
                baseline = None

            # Transcript
            transcript_path = os.path.join("../data/librispeech-pc-test-clean", sample_id, sample_id + ".txt")
            if os.path.exists(transcript_path):
                with open(transcript_path, "r") as f:
                    transcript_str = f.read().strip()
                    transcript = parse_transcript(transcript_str)
            else:
                warnings.warn(f"Transcript file not found: {transcript_path}")
                transcript = None

            # MFA
            mfa_path = os.path.join("../data/mfa", sample_id, sample_id + ".TextGrid")
            if os.path.exists(mfa_path):
                tg = TextGrid.fromFile(mfa_path)
                mfa_data = tg.getFirst("words").intervals
                mfa = parse_mfa(mfa_data)
            else:
                warnings.warn(f"MFA file not found: {mfa_path}")
                mfa = None

            # Final MFA
            final_mfa_path = os.path.join(directory, "mfa", sample_id, sample_id + ".TextGrid")
            if os.path.exists(final_mfa_path):
                tg = TextGrid.fromFile(final_mfa_path)
                final_mfa_data = tg.getFirst("words").intervals
                final_mfa = parse_mfa(final_mfa_data)
            else:
                # warnings.warn(f"Final MFA file not found: {final_mfa_path}")
                final_mfa = None

            obj = cls(
                sample_id=sample_id,
                final=parse_word_result_list(final_data),
                final_messages=parse_final_messages(final_messages_data),
                partials=parse_partials(partials_data),
                baseline=baseline,
                transcript=transcript,
                mfa=mfa,
                final_mfa=final_mfa,
            )

        return obj

    def build_alignments(self,
                         normalize_words=True,
                         align_to: Literal["final", "baseline", "mfa", "final_mfa"]="final",
                         temporal_tolerance: float=0.5,
                         accepted_tolerance: float=1.0,
                        ):
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
        elif align_to == "final_mfa":
            if self.final_mfa is None:
                raise ValueError("Final MFA is not set in the sample.")
            self._alignment_sequence = self.final_mfa
        else:
            raise ValueError("Invalid value for align_to. Use 'final', 'final_mfa', 'mfa' or 'baseline'.")
        
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
        self.wfc = [
            word_first_correct_response(i, self._alignment_sequence, self.partials, self.alignments)
            for i in range(len(self._alignment_sequence))
        ]
        return self.wfc
    
    def word_first_finals(self):
        self.wff = [
            word_first_final_response(self._alignment_sequence, self.partials, i, self.alignments)
            for i in range(len(self._alignment_sequence))
        ]
        return self.wff
    
    def word_error_rate(self):
        if self._wer is not None:
            return self._wer.word_error_rate()
        
        if self._alignment_sequence is None:
            raise ValueError("Alignment sequence is not set. Call build_alignments() first.")
        
        self._wer = openwer.process_words("en", word_sequence_to_string(self.mfa), word_sequence_to_string(self.final))
        return self._wer.word_error_rate()
