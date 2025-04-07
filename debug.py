import logging
import asyncio
from jiwer import wer

from src.eval.Dataset import Dataset
from src.eval.StreamingTranscriber import StreamingTranscriber
from src.eval.DebugTranscriberAdapter import DebugTranscriberAdapter
from src.eval.WebsocketTranscriberAdapter import WebsocketTranscriberAdapter

from src.melvin.Transcriber import Transcriber as WhisperTranscriber
from src.melvin.WhisperStreamingTranscriberAdapter import WhisperStreamingTranscriberAdapter

from src.helper.logging import init_logger, set_global_loglevel

init_logger()
set_global_loglevel("INFO")

logger = logging.getLogger("src.Main")

dataset = Dataset()
element = next(dataset)

logger.info(f"Evaluating dataset element {element[0]} with length {len(element[1])} bytes ")

w = WhisperTranscriber.for_gpu("large-v3-turbo", [0])
adapter = WhisperStreamingTranscriberAdapter(w)

transcriber = StreamingTranscriber(adapter, chunk_length_ms=1000)


async def run_transcription():
    transcription = await transcriber.transcribe(element[1])
    logger.info(f'Transcription result: "{transcription}"')
    transcription_wer = wer(element[2], transcription)
    logger.info(f"Transcription WER: {transcription_wer}")
    offline_transcription, _ = w.transcribe(element[1])
    offline_transcription = " ".join([s.text for s in offline_transcription])
    logger.info(f"Offline Transcription result: {offline_transcription}")
    offline_wer = wer(element[2], offline_transcription)
    logger.info(f"Offline Transcription WER: {offline_wer}")


asyncio.run(run_transcription())
