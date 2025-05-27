from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.eval.PredictionAlignment import PredictionAlignment
    from src.eval.SampleResult import WordResult, PartialResult


def word_first_final_response(
    alignment_sequence: list["WordResult"],
    partials: list["PartialResult"],
    alignment_index,
    prediction_alignments: list["PredictionAlignment"],
):
    """
    Determine the first partial after which the final word at final_index
    no longer changes and can be considered stable.
    """
    target_word = alignment_sequence[alignment_index]
    word_start = target_word.start

    candidate = None
    last_match_index = -2  # ensure detection of initial match without false "gap"

    for partial_alignment in prediction_alignments:
        alignment = next(
            filter(
                lambda wa: wa.final_word_index == alignment_index,
                partial_alignment.accepted_alignments,
            ),
            None,
        )
        if alignment is None:
            continue
        partial = partials[alignment.partial_timestep]
        if alignment.partial_timestep > last_match_index + 1:
            candidate = {
                "timestep": alignment.partial_timestep,
                "observation_time": partial.observation_time,
                "latency": partial.observation_time - word_start,
            }
        last_match_index = alignment.partial_timestep

    return candidate
