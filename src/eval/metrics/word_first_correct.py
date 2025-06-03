from src.eval.utils.is_equal_word import is_equal_word

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.eval.PredictionAlignment import PredictionAlignment
    from src.eval.SampleResult import WordResult, PartialResult


def word_first_correct_response(
    control_sequence_index: int,
    control_sequence: list["WordResult"],
    partial_predictions: list["PartialResult"],
    partial_prediction_alignments: list["PredictionAlignment"],
):
    """
    Given the index of a final word, return the first partial response that contains the word
    and the time it was produced.
    """
    target_word = control_sequence[control_sequence_index]
    word_start = target_word.start

    for prediction_alignment in partial_prediction_alignments:
        alignment = next(
            filter(
                lambda wa: wa.final_word_index == control_sequence_index,
                prediction_alignment.accepted_alignments,
            ),
            None,
        )
        if alignment is None:
            continue
        partial = partial_predictions[alignment.partial_timestep]
        w = partial.result[alignment.partial_word_index]
        if is_equal_word(w, target_word, normalize=False):
            return {
                "timestep": alignment.partial_timestep,
                "observation_time": partial.observation_time,
                "latency": partial.observation_time - word_start,
            }

    return None
