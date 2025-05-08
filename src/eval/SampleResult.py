from dataclasses import dataclass
import os
import json

from src.eval.PredictionAlignment import PredictionAlignment

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
    alignments: list[PredictionAlignment] = None

    def __init__(
        self,
        sample_id: str,
        final: list[WordResult],
        final_messages: list[FinalMessage],
        partials: list[PartialResult],
    ):
        self.sample_id = sample_id
        self.final = final
        self.final_messages = final_messages
        self.partials = partials

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

        with open(os.path.join(directory, sample_id + "_final.json"), "r") as f:
            final_data = json.load(f)

        with open(
            os.path.join(directory, sample_id + "_final_messages.json"), "r"
        ) as f:
            final_messages_data = json.load(f)

        with open(os.path.join(directory, sample_id + "_partial.json"), "r") as f:
            partials_data = json.load(f)

        obj = cls(
            sample_id=sample_id,
            final=parse_word_result_list(final_data),
            final_messages=parse_final_messages(final_messages_data),
            partials=parse_partials(partials_data),
        )

        return obj

    def build_alignments(self, normalize_words=True):
        self.alignments = []
        for timestep in range(len(self.partials)):
            alignment = PredictionAlignment(self, timestep, normalize_words=normalize_words).build()
            self.alignments.append(alignment)
        return self