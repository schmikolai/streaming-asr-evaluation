import logging
import asyncio

from src.eval.Dataset import Dataset
from src.eval.StreamingTranscriber import StreamingTranscriber
from src.eval.Runner import Runner

from src.melvin.Transcriber import Transcriber as WhisperTranscriber
from src.melvin.WhisperStreamingTranscriberAdapter import WhisperStreamingTranscriberAdapter

from src.helper.write_result import filename_from_setup, write_result
from src.helper.logging import init_logger
from src.helper.config import CONFIG

init_logger()

logger = logging.getLogger("src.Main")

experiment = CONFIG["experiment"]

dataset = Dataset(experiment.get("dataset", "librispeech-pc-test-clean"),)

w = WhisperTranscriber.for_gpu(experiment["model"], [0])
adapter = WhisperStreamingTranscriberAdapter(w,
                                             transcription_trigger_threshold_seconds=float(experiment["transcription_interval"]),
                                             final_transcription_threshold=int(experiment["final_transcription_threshold"]),
                                             final_publish_threshold_seconds=float(experiment["final_publish_threshold_seconds"]))

transcriber = StreamingTranscriber(adapter, chunk_length_ms=int(experiment["transcription_interval"]*1000))

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
