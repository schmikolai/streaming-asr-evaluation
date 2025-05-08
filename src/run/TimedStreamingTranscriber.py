from src.melvin.stream import Stream
from src.helper.byte_iterator import iter_chunks
from src.eval.OutputHandler import OutputHandler
from tqdm import tqdm
import time
import asyncio

import logging
logger = logging.getLogger(__name__)

class TimedStreamingTranscriber:
    def __init__(self,
                 stream: Stream,
                 output_handler: OutputHandler,
                 sample_rate: int = 16000,
                 chunk_length_ms: int = 100,
                 frame_bit_size: int = 16):
        """
        Initializes the StreamingTranscriber with the given transcriber adapter.
        Args:
            stream (Stream): The melvin stream to use for transcribing audio data.
            sample_rate (int): The sample rate of the audio data. Defaults to 16000.
            chunk_length_ms (int): The length of each chunk in milliseconds. Defaults to 100.
            frame_bit_size (int): The size of each frame in bytes. Defaults to 16.
        """
        self.stream = stream
        self.output_handler = output_handler
        self.chunk_length_ms = chunk_length_ms
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
        print(f"Transcribing audio data with {len(audio_bytes)} bytes")
        
        start_time = time.perf_counter()
        interval = self.chunk_length_ms / 1000
        tasks = []

        self.output_handler.init_timer(offset=-interval)

        # for idx, chunk in enumerate(tqdm(iter_chunks(audio_bytes, self.chunk_size), total=len(audio_bytes) // self.chunk_size, desc="Transcribing", unit="chunk")):
        for idx, chunk in enumerate(iter_chunks(audio_bytes, self.chunk_size)):
            task = asyncio.create_task(self.stream.receive_bytes(chunk))
            tasks.append(task)
            
            # Berechne geplante nächste Zeit
            next_time = start_time + (idx + 1) * interval
            sleep_duration = next_time - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

        await asyncio.gather(*tasks)
        self.stream.end_stream()
        return ""
        

