import time
from typing import Dict, List
import traceback
import logging

from faster_whisper.transcribe import Word

from src.helper.local_agreement import LocalAgreement
from src.melvin.Transcriber import Transcriber
from src.run.BaseTranscriberAdapter import BaseTranscriberAdapter

# To Calculate the seconds of audio in a chunk of 16000 Hz, 2 bytes per sample and 1 channel (as typically used in Whisper):
# 16000 Hz * 2 bytes * 1 channel = 32000 bytes per second
DEFAULT_BYTES_PER_SECOND = 32000

# This is the number of words we wait before printing a final transcription
DEFAULT_FINAL_TRANSCRIPTION_THRESHOLD = 6

# Max size of window defined in bytes
MAX_WINDOW_SIZE_BYTES = DEFAULT_BYTES_PER_SECOND * 15

# Bytes after which a retranscription of the window is triggered
DEFAULT_PARTIAL_TRANSCRIPTION_THRESHOLD_SECONDS = 1.0

# If no final has been published for this long just publish all as final
# This is mostly for cases where no audio data is sent
DEFAULT_FINAL_PUBLISH_SECOND_THRESHOLD = 5.0


class WhisperStreamingTranscriberAdapter(BaseTranscriberAdapter):
    def __init__(
        self,
        transcriber: Transcriber,
        bytes_per_second: int = 32000,
        transcription_trigger_threshold_seconds: float = 1.0,
        final_transcription_threshold: int = 6,
        final_publish_threshold_seconds: float = 5.0,
        max_window_size_seconds: float = 15.0,
    ):
        """
        `TranscriberAdapter` implementeation for the Whisper model. It handles the streaming of audio data and running the model with a sliding window approach.
        Args:
            transcriber (Transcriber): The transcriber to use for transcription.
            bytes_per_second (int): The number of bytes per second of audio data.
            transcription_trigger_threshold_seconds (float): The number of seconds after which a transcription is triggered.
            final_transcription_threshold (int): The number of words after which a final transcription is printed.
            final_publish_threshold_seconds (float): The number of seconds after which a final transcription is forcibly published.
            max_window_size_seconds (float): The maximum size of the sliding window in seconds.
        """
        self.logger = logging.getLogger(__name__)
        self.transcriber = transcriber
        self.bytes_per_second = bytes_per_second
        self.transcription_trigger_threshold_byte = int(transcription_trigger_threshold_seconds * bytes_per_second)
        self.final_transcription_threshold = final_transcription_threshold
        self.final_publish_threshold_byte = int(final_publish_threshold_seconds * bytes_per_second)
        self.max_window_size_bytes = int(max_window_size_seconds * bytes_per_second)

    def _start_stream(self) -> bool:
        self.logger.debug("Starting stream")
        self.sliding_window = b""
        self.total_bytes = b""
        self.window_start_timestamp = 0
        self.agreement = LocalAgreement()
        self.bytes_received_since_last_transcription = 0
        self.final_transcriptions = []
        self.previous_byte_count = 0

        self.last_transcription_timestamp = time.time()
        self.last_final_published = time.time()
        return True

    def _close_stream(self):
        self.transcribe_sliding_window(self.sliding_window)
        self.flush_final()

    async def transcribe_bytes(self, audio_bytes: bytes) -> str | None:
        self.bytes_received_since_last_transcription += len(audio_bytes)
        self.sliding_window += audio_bytes
        self.total_bytes += audio_bytes

        if self.bytes_received_since_last_transcription >= self.transcription_trigger_threshold_byte:
            self.bytes_received_since_last_transcription = 0
            self.last_transcription_timestamp = time.time()

            self.transcribe_sliding_window(self.sliding_window)

        # Send final if either threshold is reached or sentence ended
        if (
            self.agreement.get_confirmed_length() > self.final_transcription_threshold
            or self.agreement.contains_has_sentence_end()
            or (time.time() - self.last_final_published) * self.bytes_per_second >= self.final_publish_threshold_byte
        ):
            self.logger.debug(f"NEW FINAL: length of chunk cache: {len(self.sliding_window)}")
            finals = self.flush_final()
            if finals is not None:
                self.logger.debug(f"Final transcription: {finals['text']}")
                return finals["text"]

    def finalize_transcript(self) -> Dict:
        current_transcript = self.agreement.unconfirmed
        return self.build_result_from_words(current_transcript)

    def flush_final(self) -> Dict:
        """Function to send a final to the client and update the content on the sliding window"""
        try:
            agreed_results = []
            if self.agreement.contains_has_sentence_end():
                agreed_results = self.agreement.flush_at_sentence_end()
            else:
                agreed_results = self.agreement.flush_confirmed()

            result = self.build_result_from_words(agreed_results)
            # The final did not contain anything to send
            if len(result["result"]) == 0:
                return
            self.final_transcriptions.append(result)

            # Shorten window if needed
            if len(self.sliding_window) > self.max_window_size_bytes:
                bytes_to_cut_off = len(self.sliding_window) - self.max_window_size_bytes
                self.logger.debug(f"Reducing sliding window size by {bytes_to_cut_off} bytes")
                self.previous_byte_count += bytes_to_cut_off
                self.window_start_timestamp += bytes_to_cut_off / self.bytes_per_second
                self.sliding_window = self.sliding_window[bytes_to_cut_off:]

            self.logger.debug(f"Published final of {len(agreed_results)}.")
            self.last_final_published = time.time()

            return result

        except Exception:
            self.logger.error(f"Error while transcribing audio: {traceback.format_exc()}")

    def build_result_from_words(self, words: List[Word], save=True) -> Dict:
        overall_transcribed_seconds = self.previous_byte_count / self.bytes_per_second

        cutoff_timestamp = 0
        if len(self.final_transcriptions) > 0:
            cutoff_timestamp = self.final_transcriptions[-1]["result"][-1]["end"]

        result = {"result": [], "text": ""}
        for word in words:
            start = float(f"{word.start:.6f}") + overall_transcribed_seconds
            end = float(f"{word.end:.6f}") + overall_transcribed_seconds
            conf = float(f"{word.probability:.6f}")
            if end <= cutoff_timestamp + 0.01 and save:
                continue
            result["result"].append(
                {
                    "conf": conf,
                    # The start time and end time is the time of the word minus the time of the current final
                    "start": start,
                    "end": end,
                    "word": word.word.strip(),
                }
            )
        result["text"] = " ".join([x["word"] for x in result["result"]])
        return result

    def transcribe_sliding_window(self, window_content) -> str:
        if len(window_content) == 0:
            self.logger.warning("Received empty chunk, skipping transcription.")
            return  # Skip transcription for empty chunk

        try:
            start_time = time.time()
            self.bytes_received_since_last_transcription = 0

            # Pass the chunk to the transcriber
            segments, _ = self.transcriber.transcribe(
                window_content,
            )

            cutoff_timestamp = 0
            if len(self.final_transcriptions) > 0:
                # Absolute timestamp - thrown out bytes -> timestamp in the current window
                cutoff_timestamp = self.final_transcriptions[-1]["result"][-1]["end"] - (
                    self.previous_byte_count / self.bytes_per_second
                )

            new_words = []

            for segment in list(segments):
                if segment.words is None:
                    continue
                for word in segment.words:
                    new_words.append(word)

            text = ""

            if len(self.agreement.unconfirmed) > 0:
                # Hacky workaround for doubled word between finals
                if len(new_words) > 0 and len(self.final_transcriptions) > 0:
                    if new_words[0].word == self.final_transcriptions[-1]["result"][-1]["word"]:
                        new_words.pop(0)
                text = " ".join([w.word for w in new_words if w.end > (cutoff_timestamp + 0.01)])

            self.agreement.merge(new_words)

            end_time = time.time()
            self.logger.debug("Partial transcription took {:.2f} s".format(end_time - start_time))

            # adjust time between transcriptions
            # self.update_partial_threshold(end_time - start_time)

            return text

        except Exception:
            self.logger.error("Error while transcribing audio: {}".format(traceback.format_exc()))

    def update_partial_threshold(self, last_run_duration: float):
        # dont adjust any timings with a small window
        # these adjustments would be overwritten anyway
        if (
            len(self.sliding_window) < self.max_window_size_bytes * 0.75
            and last_run_duration < self.transcription_trigger_threshold_byte / self.bytes_per_second
        ):
            self.logger.info(
                f"Current window too small for adjustment ({len(self.sliding_window)}/{self.max_window_size_bytes * 0.75})"
            )
            return
        new_threshold = (last_run_duration * self.bytes_per_second) + 0.5
        self.logger.info(f"Adjusted threshold duration to : {new_threshold / self.bytes_per_second}")
        self.transcription_trigger_threshold_byte = new_threshold

    def final_transcript(self):
        """
        Returns the final transcript after all audio data has been processed.
        """
        if len(self.final_transcriptions) == 0:
            return ""
        return " ".join(map(lambda x: x["text"], self.final_transcriptions))
