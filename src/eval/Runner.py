from src.eval.StreamingTranscriber import StreamingTranscriber
from src.eval.Dataset import Dataset

import jiwer
import time
import logging

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, transcriber: StreamingTranscriber, dataset: Dataset):
        self.transcriber = transcriber
        self.dataset = dataset
        self.results = dict()

    async def run(self):
        for id, audio_bytes, transcription in self.dataset:
            start_time = time.time()
            pred_transcription = await self.transcriber.transcribe(audio_bytes)
            end_time = time.time()
            wer = jiwer.wer(transcription, pred_transcription)
            logger.info(f"Transcribed element {id} with WER {wer}")
            self.results[id] = {
                "wer": wer,
                "pred_transcription": pred_transcription,
                "time": end_time - start_time
            }
