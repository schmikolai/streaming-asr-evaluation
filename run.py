import logging
import asyncio

from src.eval.Dataset import Dataset
from src.eval.StreamingTranscriber import StreamingTranscriber
from src.eval.Runner import Runner

from src.melvin.Transcriber import Transcriber as WhisperTranscriber
from src.melvin.WhisperStreamingTranscriberAdapter import WhisperStreamingTranscriberAdapter

from src.helper.write_result import filename_from_setup, write_result
from src.helper.logging import init_logger

init_logger()

logger = logging.getLogger("src.Main")

dataset = Dataset()

w = WhisperTranscriber.for_gpu("large-v3-turbo", [0])
adapter = WhisperStreamingTranscriberAdapter(w)

transcriber = StreamingTranscriber(adapter, chunk_length_ms=1000)

filename = filename_from_setup(
    dataset,
    transcriber,
    adapter
)

logger.info(filename)

async def run():
    runner = Runner(transcriber, dataset)
    await runner.run()
    write_result(filename, runner.results)


asyncio.run(run())
