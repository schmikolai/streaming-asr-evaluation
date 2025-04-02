from src.Dataset import Dataset
from src.DebugTranscriberAdapter import DebugTranscriberAdapter
from src.StreamingTranscriber import StreamingTranscriber
from src.helper.logging import init_logger

init_logger()

import logging
logger = logging.getLogger("Main")

dataset = Dataset()
element = next(dataset)

logger.debug(f"Dataset element {element[0]} with length {len(element[1])} bytes and transcript: \"{element[2]}\"")

transcriber = StreamingTranscriber(DebugTranscriberAdapter())
logger.info(f"Transcribing dataset element with ID {element[0]}")
transcriber.transcribe(element[1])

