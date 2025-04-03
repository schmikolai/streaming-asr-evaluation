from src.eval.BaseTranscriberAdapter import BaseTranscriberAdapter
import numpy as np
import logging
logger = logging.getLogger(__name__)

def bytes_to_array(audio_bytes: bytes) -> np.array:
    frames_int = np.frombuffer(audio_bytes, dtype=np.int16)
    frames_float = frames_int.astype(np.float16) / 32767.0
    return frames_float

class DebugTranscriberAdapter(BaseTranscriberAdapter):
    def _start_stream(self) -> bool:
        logger.debug("Starting debug transcriber stream")
        self.chunk_count = 0
        return True
    
    def _close_stream(self) -> bool:
        logger.info(f"Transcribed {self.chunk_count} audio chunks")
        return True

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        transcription = f"Audio chunk {self.chunk_count} of length {len(audio_bytes)} bytes"
        if self.chunk_count % 100 == 0:
            logger.debug(transcription)
            logger.debug(bytes_to_array(audio_bytes))

        self.chunk_count += 1
        return transcription