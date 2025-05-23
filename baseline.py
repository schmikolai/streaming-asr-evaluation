import logging
import asyncio

from src.run.Dataset import Dataset
from src.run.OfflineTranscriber import OfflineTranscriber
from src.run.Runner import Runner

from src.melvin.Transcriber import Transcriber as WhisperTranscriber
from src.melvin.WhisperStreamingTranscriberAdapter import WhisperStreamingTranscriberAdapter

from src.helper.write_result import filename_from_setup, write_result
from src.helper.logger import init_logger
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
    runner = Runner(transcriber, dataset, out_file=filename)
    await runner.run()

asyncio.run(run())
