import logging
import asyncio

from src.run.Dataset import Dataset
from src.run.RealtimeRunner import RealtimeRunner

from src.melvin.StreamTranscriber import StreamTranscriber

from src.helper.write_result import outdir_from_setup
from src.helper.logger import init_logger, set_global_loglevel
from src.helper.config import CONFIG

from dotenv import load_dotenv

load_dotenv()

init_logger()

log_level = CONFIG.get("log_level", None)

if log_level is not None:
    set_global_loglevel(log_level)

logger = logging.getLogger("src.Main")

experiment = CONFIG["experiment"]

logger.info(f"Running experiment {experiment}")

dataset = Dataset(experiment.get("dataset", "librispeech-pc-test-clean"), dataset_ids=experiment.get("dataset_ids", None))

method = experiment.get("method", "melvin")

if method not in ["melvin", "assemblyai"]:
    raise ValueError(f"Method {method} is not supported. Choose 'melvin' or 'assemblyai'.")

w = None
if method == "melvin":
    w = StreamTranscriber.for_gpu(experiment["model"], [0])

outdir = outdir_from_setup(
    dataset,
    w
)

logger.info(outdir)

async def run():
    runner = RealtimeRunner(dataset, method=method, stream_transcriber=w, out_dir=outdir)
    await runner.run()


asyncio.run(run())
