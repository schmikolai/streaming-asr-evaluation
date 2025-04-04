from src.eval.BaseTranscriberAdapter import BaseTranscriberAdapter
from src.helper.byte_iterator import iter_chunks
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

class StreamingTranscriber:
    def __init__(self,
                 transcriber_adapter: BaseTranscriberAdapter,
                 sample_rate: int = 16000,
                 chunk_length_ms: int = 500,
                 frame_bit_size: int = 16):
        """
        Initializes the StreamingTranscriber with the given transcriber adapter.
        Args:
            transcriber_adapter (TranscriberAdapter): The adapter to use for transcribing audio data.
            sample_rate (int): The sample rate of the audio data. Defaults to 16000.
            chunk_length_ms (int): The length of each chunk in milliseconds. Defaults to 100.
            frame_bit_size (int): The size of each frame in bytes. Defaults to 2.
        """
        self.transcriber_adapter = transcriber_adapter
        self.chunk_size = sample_rate * chunk_length_ms // 1000 * frame_bit_size // 8

    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes the given audio bytes as stream, by segmenting the audio into chucks and continuously sending new bytes to the TranscriberAdapter.
        Args:
            audio_bytes (bytes): Whole audio data to be transcribed.
        Returns:
            str: The transcribed text from the audio data.
        """
        logger.info(f"Transcribing audio data with {len(audio_bytes)} bytes")
        logger.debug(f"Bytes per chunk: {self.chunk_size}")
        with self.transcriber_adapter:
            for chunk in tqdm(iter_chunks(audio_bytes, self.chunk_size), total=len(audio_bytes) // self.chunk_size, desc="Transcribing", unit="chunk"):
                await self.transcriber_adapter.transcribe_bytes(chunk)
        
        final_transcript = self.transcriber_adapter.get_final_transcript()
        return final_transcript
        

