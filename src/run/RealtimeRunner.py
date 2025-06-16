from src.run.TimedStreamingTranscriber import TimedStreamingTranscriber
from src.run.Dataset import Dataset
from src.run.OutputHandler import OutputHandler
from src.run.Stream import Stream
from src.melvin.StreamTranscriber import StreamTranscriber

from tqdm import tqdm
from typing import Literal
import os
import json
import logging

logger = logging.getLogger(__name__)


class RealtimeRunner:
    def __init__(
        self,
        dataset: Dataset,
        method: Literal["melvin", "assemblyai"] = "melvin",
        out_dir: str = None,
        stream_transcriber: StreamTranscriber = None,
    ):
        self.stream_transcriber = stream_transcriber
        self.dataset = dataset
        self.out_dir = out_dir
        self.method = method

    async def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for id, audio_bytes, transcription in tqdm(self.dataset):
            out = OutputHandler()
            stream = Stream.create(
                type=self.method,
                output_handler=out,
                whisper_transcriber=self.stream_transcriber,
            )
            transcriber = TimedStreamingTranscriber(stream, out, chunk_length_ms=50)
            y_pred = await transcriber.transcribe(audio_bytes)
            with open(os.path.join(self.out_dir, f"{id}_final.json"), "w") as f:
                json.dump(out.final_words, f)
            with open(
                os.path.join(self.out_dir, f"{id}_final_messages.json"), "w"
            ) as f:
                json.dump(out.final_messages, f)
            with open(os.path.join(self.out_dir, f"{id}_partial.json"), "w") as f:
                json.dump(out.partial_predictions, f)
