from src.eval.TimedStreamingTranscriber import TimedStreamingTranscriber
from src.eval.Dataset import Dataset
from src.eval.OutputHandler import OutputHandler
from src.melvin.stream import Stream
from src.melvin.StreamTranscriber import StreamTranscriber
from src.helper.write_result import write_result

from tqdm import tqdm
import os
import json
import logging

logger = logging.getLogger(__name__)

class RealtimeRunner:
    def __init__(self, stream_transcriber: StreamTranscriber, dataset: Dataset, out_dir: str = None):
        self.stream_transcriber = stream_transcriber
        self.dataset = dataset
        self.out_dir = out_dir

    async def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for id, audio_bytes, transcription in tqdm(self.dataset):
            out = OutputHandler()
            stream = Stream(self.stream_transcriber, 0, out)
            transcriber = TimedStreamingTranscriber(stream, out)
            y_pred = await transcriber.transcribe(audio_bytes)
            with open(os.path.join(self.out_dir, f"{id}_final.json"), "w") as f:
                json.dump(out.final_words, f)
                logger.info()
            with open(os.path.join(self.out_dir, f"{id}_partial.json"), "w") as f:
                json.dump(out.partial_predictions, f)
