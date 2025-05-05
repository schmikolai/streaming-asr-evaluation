import logging
import asyncio

from src.eval.Dataset import Dataset
from src.eval.RealtimeRunner import RealtimeRunner

from src.melvin.StreamTranscriber import StreamTranscriber

from src.helper.write_result import outdir_from_setup
from src.helper.logging import init_logger
from src.helper.config import CONFIG

init_logger()

logger = logging.getLogger("src.Main")

experiment = CONFIG["experiment"]

logger.info(f"Running experiment {experiment}")

dataset = Dataset(experiment.get("dataset", "librispeech-pc-test-clean"), dataset_ids=experiment.get("dataset_ids", None))

w = StreamTranscriber.for_gpu(experiment["model"], [0])

outdir = outdir_from_setup(
    dataset,
    w
)

logger.info(outdir)

async def run():
    runner = RealtimeRunner(w, dataset, outdir)
    await runner.run()


asyncio.run(run())
