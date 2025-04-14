import logging
import asyncio

from src.eval.Dataset import Dataset
from src.eval.OfflineTranscriber import OfflineTranscriber
from src.eval.Runner import Runner

from src.melvin.Transcriber import Transcriber as WhisperTranscriber
from src.melvin.WhisperStreamingTranscriberAdapter import WhisperStreamingTranscriberAdapter

from src.helper.write_result import filename_from_setup, write_result
from src.helper.logging import init_logger
from src.helper.config import CONFIG

init_logger()

logger = logging.getLogger("src.Main")

experiment = CONFIG["experiment"]

dataset = Dataset()

w = WhisperTranscriber.for_gpu(experiment["model"], [0])
transcriber = OfflineTranscriber(w)

filename = f"baseline_{experiment['model']}.csv"

logger.info(filename)

async def run():
    runner = Runner(transcriber, dataset)
    await runner.run()

asyncio.run(run())
