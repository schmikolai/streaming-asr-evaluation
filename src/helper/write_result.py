from src.eval.Dataset import Dataset
from src.eval.StreamingTranscriber import StreamingTranscriber
from src.melvin.WhisperStreamingTranscriberAdapter import WhisperStreamingTranscriberAdapter

import time
import os
import pandas as pd

def filename_from_setup(
    dataset: Dataset,
    transcriber: StreamingTranscriber,
    whisper_transcriber: WhisperStreamingTranscriberAdapter,
) -> str:
    return "{}_{}_cs{}_{}_ws{}_ft{}.csv".format(
        time.strftime("%Y-%m-%d_%H-%M-%S"),
        dataset.dataset_name,
        transcriber.chunk_size,
        whisper_transcriber.transcriber._model_name,
        whisper_transcriber.max_window_size_bytes / whisper_transcriber.bytes_per_second,
        whisper_transcriber.final_transcription_threshold,
    )

def write_result(
        file_name: str,
        result: dict,
        output_dir: str = "out",
) -> str:
    output_path = os.path.join(output_dir, file_name)
    df = pd.DataFrame.from_dict(result, orient="index")
    df.to_csv(output_path, index=True)
