from dataclasses import dataclass
from typing import TYPE_CHECKING
from src.eval.utils.is_equal_word import is_equal_word

if TYPE_CHECKING:
    from src.eval.SampleResult import SampleResult, WordResult


@dataclass
class WordAlignment:
    partial_timestep: int
    partial_word_index: int
    final_word_index: int


class PredictionAlignment:
    accepted_alignments: list[WordAlignment] = []
    potential_alignments: list[WordAlignment] = []
    confirmed_alignments: list[WordAlignment] = []

    sample: "SampleResult"
    timestep: int

    _alignment_sequence: list["WordResult"] = []

    def __init__(
        self,
        sample: "SampleResult",
        timestep: int,
        temporal_tolerance=0.5,
        accepted_tolerance=1.0,
        normalize_words=True,
    ):
        self.timestep = timestep
        self.sample = sample
        self.temporal_tolerance = temporal_tolerance
        self.accepted_tolerance = accepted_tolerance
        self.normalize_words = normalize_words

    def build(self):
        self.accepted_alignments = []
        self.potential_alignments = []
        self.confirmed_alignments = []
        self.unalignments = []

        if self.sample._alignment_sequence is None:
            raise ValueError(
                "Alignment sequence of SampleResult needs to be set before building the prediction alignment."
            )

        self._alignment_sequence = self.sample._alignment_sequence

        self._build_accepted_alignments()
        self._confirm_potential_alignments()

        return self

    def _get_search_start_index(self, partial: "WordResult"):
        if len(self.accepted_alignments) == 0:
            search_time_start = partial.start - self.temporal_tolerance
            for f_idx in range(len(self._alignment_sequence)):
                final_word = self._alignment_sequence[f_idx]
                if final_word.end > search_time_start:
                    return f_idx
            return 0
            return len(self._alignment_sequence) - 1
        last_accepted = self.accepted_alignments[-1]
        return last_accepted.final_word_index + 1

    def _get_best_alignment(self, partial: "WordResult", p_idx: int, start_index: int):
        for f_idx in range(start_index, len(self._alignment_sequence)):
            final_word = self._alignment_sequence[f_idx]
            if is_equal_word(partial, final_word, normalize=self.normalize_words):
                has_temporal_overlap = bool(min(partial.end, final_word.end) - max(partial.start, final_word.start) > -self.temporal_tolerance)
                return has_temporal_overlap, WordAlignment(self.timestep, p_idx, f_idx)
            if final_word.start > partial.end + self.temporal_tolerance:
                break

        best_alignment = None
        best_alignment_temporal_overlap = 0

        for f_idx in range(start_index, len(self._alignment_sequence)):
            final_word = self._alignment_sequence[f_idx]
            if final_word.start > partial.end:
                break
            temporal_overlap = min(partial.end, final_word.end) - max(partial.start, final_word.start)
            if temporal_overlap > best_alignment_temporal_overlap:
                best_alignment_temporal_overlap = temporal_overlap
                best_alignment = WordAlignment(self.timestep, p_idx, f_idx)

        if best_alignment_temporal_overlap > 0:
            return False, best_alignment

        return False, None

    def _build_accepted_alignments(self):
        for p_idx, word in enumerate(self.sample.partials[self.timestep].result):
            i = self._get_search_start_index(word)
            accepted, word_alignment = self._get_best_alignment(word, p_idx, i)
            if accepted:
                self.accepted_alignments.append(word_alignment)
            elif word_alignment is not None:
                self.potential_alignments.append(word_alignment)

    def _confirm_potential_alignments(self):
        used_final_indices = [wa.final_word_index for wa in self.accepted_alignments]
        self.confirmed_alignments = [wa for wa in self.accepted_alignments]
        for wa in self.potential_alignments:
            if wa.final_word_index not in used_final_indices:
                self.confirmed_alignments.append(wa)
            else:
                self.unalignments.append(wa)
