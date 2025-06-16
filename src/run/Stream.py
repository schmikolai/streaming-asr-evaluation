
from src.melvin.StreamTranscriber import StreamTranscriber as WhisperTranscriber
from src.run.OutputHandler import OutputHandler
from typing import Literal

class Stream:
    def receive_bytes(self, chunk: bytes):
        """
        Simulates receiving a chunk of audio data.
        Args:
            chunk (bytes): The audio data chunk to be processed.
        """
        # This method should be implemented in the actual Stream class
        raise NotImplementedError("This method should be implemented in the actual Stream class.")
    

    def start_stream(self):
        """
        Simulates starting the audio stream.
        """
        # This method should be implemented in the actual Stream class
        raise NotImplementedError("This method should be implemented in the actual Stream class.")
    
    def end_stream(self):
        """
        Simulates ending the audio stream.
        """
        # This method should be implemented in the actual Stream class
        raise NotImplementedError("This method should be implemented in the actual Stream class.")
    

    @classmethod
    def create(cls,
               type: Literal["melvin", "assemblyai"],
               output_handler: OutputHandler,
               whisper_transcriber: WhisperTranscriber = None):
        """
        Creates a new instance of the Stream class.
        """
        if type == "melvin":
            from src.melvin.stream import Stream as MelvinStream
            return MelvinStream(whisper_transcriber, 0, output_handler)
        elif type == "assemblyai":
            from src.run.AssemblyAIStream import AssemblyAIStream
            return AssemblyAIStream(output_handler)
        else:
            raise ValueError(f"Unknown stream type: {type}")