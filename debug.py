from src.eval.Dataset import Dataset
from src.helper.logging import init_logger
from src.eval.StreamingTranscriber import StreamingTranscriber

from src.melvin.stream_transcriber import Transcriber as Whisper
from src.melvin.stream import Stream

from src.eval.DebugTranscriberAdapter import DebugTranscriberAdapter
from src.eval.WebsocketTranscriberAdapter import WebsocketTranscriberAdapter

import asyncio

init_logger()

import logging
logger = logging.getLogger("src.Main")

dataset = Dataset()
element = next(dataset)

logger.info(f"Dataset element {element[0]} with length {len(element[1])} bytes and transcript: \"{element[2]}\"")

w = Whisper.for_gpu("large-v3-turbo", [0])
adapter = Stream(w)

transcriber = StreamingTranscriber(adapter, chunk_length_ms=200)
logger.info(f"Transcribing dataset element with ID {element[0]}")

async def run_transcription():
    transcription = await transcriber.transcribe(element[1])
    logger.info(f"Transcription result: \"{transcription}\"")

asyncio.run(run_transcription())

