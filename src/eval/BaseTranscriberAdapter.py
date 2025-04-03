import logging
logger = logging.getLogger(__name__)

class BaseTranscriberAdapter:
    """
    A base adapter class for transcribing audio data. This class is intended to be 
    subclassed to provide specific implementations for transcribing audio bytes.
    """

    def __enter__(self):
        """Starts the byte stream when entering the context."""
        self._start_stream()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stops the stream when exiting the context."""
        self._close_stream()

    def _start_stream(self) -> bool:
        """
        Starts the byte stream for the transcriber. This method should be called before transcribing audio data.
        Returns:
            bool: True if the byte stream was successfully started, False otherwise.
        """
        logger.warning("TranscriberAdapter implementation did not provide a _start_stream method. Nothing to start.")
        return True

    def _close_stream(self) -> bool:
        """
        Closes the byte stream for the transcriber. This method should be called after transcribing audio data.
        Returns:
            bool: True if the byte stream was successfully closed, False otherwise.
        """
        logger.warning("TranscriberAdapter implementation did not provide a _close_stream method. Nothing to close.")
        return True

    async def transcribe_bytes(self, audio_bytes: bytes) -> str | None:
        """
        Transcribes the stream, given the new audio data in bytes format into text. Audio data is expected to be 16 kHz, 16-bit float PCM [-1,1].
        Args:
            audio_bytes (bytes): The audio data to be transcribed, provided as a byte stream.
        Returns:
            str: The transcribed text from the audio data.
        """
        logger.warning("TranscriberAdapter implementation did not provide a transcribe_bytes method. Cannot transcribe.")
        raise NotImplementedError
    
    def get_final_transcript(self) -> str:
        """
        Returns the final transcript after all audio data has been processed.
        """
        raise NotImplementedError