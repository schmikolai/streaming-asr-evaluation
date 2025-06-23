import asyncio
import queue
import threading
from src.run.OutputHandler import OutputHandler
from src.run.Stream import Stream
from src.helper.format_assemblyai_words import result_from_turn
import os

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

import logging
logger = logging.getLogger(__name__)

ENDPOINT = "wss://streaming.assemblyai.com/v3/ws?sample_rate=16000"
WAIT_TIMEOUT = 0.05  # seconds

class AssemblyAIStream(Stream):
    def __init__(self, output_handler: OutputHandler):
        self.output_handler = output_handler
        self.begin_event = threading.Event()
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.websocket_task = threading.Thread(target=self.run_websocket_tasks)

        api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise ValueError(" Please set AssemblyAI API key as ASSEMBLYAI_API_KEY environment variables.")

        self.client = StreamingClient(
            StreamingClientOptions(
                api_key=api_key,
                api_host="streaming.assemblyai.com",
            )
        )

        self.client.on(StreamingEvents.Begin, self.on_begin)
        self.client.on(StreamingEvents.Turn, self.on_turn)
        self.client.on(StreamingEvents.Termination, self.on_terminated)
        self.client.on(StreamingEvents.Error, self.on_error)

    async def receive_bytes(self, chunk: bytes):
        """
        Simulates receiving a chunk of audio data.
        Args:
            chunk (bytes): The audio data chunk to be processed.
        """
        logger.debug(f"Received chunk of size: {len(chunk)} bytes")
        self.audio_queue.put(chunk)

    def start_stream(self):
        """Starts the WebSocket client and begins processing audio data."""
        if not self.websocket_task.is_alive():
            logger.debug("Connecting WebSocket client...")
            self.client.connect(
                StreamingParameters(
                    sample_rate=16000,
                    format_turns=True,
                )
            )
            logger.debug("WebSocket client connected.")
            self.websocket_task.start()
            if not self.begin_event.wait(timeout=10.0):
                logger.error("Timeout waiting for WebSocket client to start.")
        else:
            logger.warning("WebSocket task is already running.")

    def end_stream(self):
        logger.debug("Ending stream...")
        self.stop_event.set()
        logger.debug("Waiting for WebSocket task to finish...")
        self.websocket_task.join()
        logger.debug("Disconnecting WebSocket client...")
        self.client.disconnect(terminate=True)
        logger.debug("Stopping event loop...")
        self.loop.call_soon_threadsafe(self.loop.stop)

    async def send_audio_with_client(self):
        """Sends the input file to the WebSocket server and prints responses."""
        try:
            data = self.audio_queue.get(timeout=WAIT_TIMEOUT)
            logger.debug(f"Sending chunk of size: {len(data)} bytes")
            self.client.stream(data)
        except queue.Empty:
            # print("Queue is empty.")
            return

    def on_begin(self, c, event: BeginEvent):
        logger.info(f"Session started: {event.model_dump_json()}")
        self.begin_event.set()

    def on_turn(self, c, turn_event: TurnEvent):
        """Receives the response from the WebSocket server."""
        # print("Receiving from WebSocket server...")
        logger.info(f"Turn: {turn_event.model_dump_json()}")
        result = result_from_turn(turn_event)
        if turn_event.turn_is_formatted:
            self.output_handler.send_final(result["result"])
        else:
            self.output_handler.send_partial(result, 0.0, result["result"][-1]["end"] if len(result["result"]) else None)

    def on_terminated(self, c, event: TerminationEvent):
        logger.info(
            f"Session terminated: {event.audio_duration_seconds} seconds of audio processed"
        )

    def on_error(self, c, error: StreamingError):
        logger.error(f"Error occurred: {error}")

    def run_websocket_tasks(self):
        """Manages WebSocket tasks."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.websocket_tasks())

    async def websocket_tasks(self):
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            await asyncio.create_task(self.send_audio_with_client())
        await asyncio.sleep(2.0)
