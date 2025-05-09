from src.run.BaseTranscriberAdapter import BaseTranscriberAdapter
from src.helper.byte_iterator import iter_chunks
from src.melvin.Transcriber import Transcriber
import logging
logger = logging.getLogger(__name__)

class OfflineTranscriber:
    def __init__(self,
                 whisper_transcriber: Transcriber
    ):
        self._whisper = whisper_transcriber

    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes the given audio bytes as stream, by segmenting the audio into chucks and continuously sending new bytes to the TranscriberAdapter.
        Args:
            audio_bytes (bytes): Whole audio data to be transcribed.
        Returns:
            str: The transcribed text from the audio data.
        """
        logger.info(f"Transcribing audio data with {len(audio_bytes)} bytes")
        data, _ = self._whisper.transcribe(audio_bytes)

        final_transcript = ""
        words = []

        for segment in data:
            words += segment.words
            # Check if the segment is a final transcription
            if segment.text:
                # Append the final transcription to the final transcript
                final_transcript += segment.text

        return final_transcript, words
        

