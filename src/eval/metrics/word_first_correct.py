from src.eval.utils.is_equal_word import is_equal_word

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.eval.PredictionAlignment import PredictionAlignment
    from src.eval.SampleResult import WordResult, PartialResult


def word_first_correct_response(
    final: list["WordResult"],
    partials: list["PartialResult"],
    final_index: int,
    prediction_alignments: list["PredictionAlignment"],
):
    """
    Given the index of a final word, return the first partial response that contains the word
    and the time it was produced.
    """
    target_word = final[final_index]
    word_start = target_word.start

    for prediction_alignment in prediction_alignments:
        alignment = next(
            filter(
                lambda wa: wa.final_word_index == final_index,
                prediction_alignment.accepted_alignments,
            ),
            None,
        )
        if alignment is None:
            continue
        partial = partials[alignment.partial_timestep]
        for w in partial.result:
            if is_equal_word(w, target_word):
                return {
                    "timestep": alignment.partial_timestep,
                    "observation_time": partial.observation_time,
                    "latency": partial.observation_time - word_start,
                }

    return None
