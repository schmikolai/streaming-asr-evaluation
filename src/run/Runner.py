from src.run.StreamingTranscriber import StreamingTranscriber
from src.run.Dataset import Dataset
from src.helper.write_result import write_result

from tqdm import tqdm
import jiwer
import time
import logging
import os
import json

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, transcriber: StreamingTranscriber, dataset: Dataset, out_file: str = None):
        self.transcriber = transcriber
        self.dataset = dataset
        self.results = dict()
        self.out_file = out_file

    async def run(self):
        for id, audio_bytes, transcription in tqdm(self.dataset):
            start_time = time.time()
            pred_transcription, data = await self.transcriber.transcribe(audio_bytes)
            end_time = time.time()
            word_dict_array = []
            for word in data:
                start = float(f"{word.start:.6f}")
                end = float(f"{word.end:.6f}")
                conf = float(f"{word.probability:.6f}")
                word_dict_array.append(
                    {
                    "conf": conf,
                    # The start time and end time is the time of the word minus the time of the current final
                    "start": start,
                    "end": end,
                    "word": word.word.strip(),
                })
            with open(f"out/baseline/{id}.json", "w") as f:
                json.dump(word_dict_array, f)
            continue
            wer = jiwer.wer(transcription, pred_transcription)
            logger.info(f"Transcribed element {id} with WER {wer}")
            self.results[id] = {
                "wer": wer,
                "pred_transcription": pred_transcription,
                "time": end_time - start_time
            }
            if self.out_file is not None:
                write_result(self.out_file, self.results)
