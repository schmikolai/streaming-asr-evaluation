from src.eval.BaseTranscriberAdapter import BaseTranscriberAdapter
import websockets
import asyncio
import threading
import queue
import logging
import json

logger = logging.getLogger(__name__)

AUDIO_FILE_LENGTH=0.1

class WebsocketTranscriberAdapter(BaseTranscriberAdapter):
    def __init__(self, websocket_url):
        self.websocket_url = websocket_url
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.loop = asyncio.new_event_loop()
        self.websocket_task = threading.Thread(target=self.run_websocket_tasks)

    def _start_stream(self) -> bool:
        self.websocket_task.start()
        self.audio_queue.queue.clear()
        self.stop_event.clear()
        self.final_result = ""
        self.chunk_size = None
        self.result_id = None
        logger.debug(f"Starting websocket stream to {self.websocket_url}")
        return True
    
    def _close_stream(self) -> bool:
        self.stop_event.set()
        self.websocket_task.join()
        return True

    async def send_file_as_websocket(self, websocket):
        """Sends the input file to the WebSocket server and prints responses."""
        try:
            data = self.audio_queue.get(timeout=AUDIO_FILE_LENGTH/10)
            await websocket.send(data)
        except queue.Empty:
            return

    async def receive_from_websocket(self, websocket):
        """Receives the response from the WebSocket server."""
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=AUDIO_FILE_LENGTH/10)
            final_result = json.loads(response).get("text", None)
            if final_result:
                self.final_result += " " + final_result
        except asyncio.TimeoutError:
            return

    def run_websocket_tasks(self):
        """Manages WebSocket tasks."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.websocket_tasks())

    async def websocket_tasks(self):
        async with websockets.connect(self.websocket_url, logger=None) as websocket:
            while not self.stop_event.is_set():
                send_task = asyncio.create_task(self.send_file_as_websocket(websocket))
                receive_task = asyncio.create_task(self.receive_from_websocket(websocket))
                await asyncio.gather(send_task, receive_task)
            logger.debug("Gracefully stopping websocket stream")
            await websocket.send("eof")

    async def transcribe_bytes(self, audio_bytes) -> str:
        if self.chunk_size is None:
            self.chunk_size = len(audio_bytes)
            logger.debug(f"Chunk size set to {self.chunk_size}")
        elif len(audio_bytes) != self.chunk_size:
            logger.warning(f"Chunk size mismatch: expected {self.chunk_size}, got {len(audio_bytes)}.")
            if len(audio_bytes) > self.chunk_size:
                audio_bytes = audio_bytes[:self.chunk_size]
            else:
                audio_bytes += b'\x00' * (self.chunk_size - len(audio_bytes))
        self.audio_queue.put(audio_bytes)
        await asyncio.sleep(AUDIO_FILE_LENGTH)
        response = self.final_result
        logger.debug(f"Received response from websocket server: '{response}'")
        return response
    
    def get_final_transcript(self) -> str:
        """
        Returns the final transcript.
        """
        return self.final_result.strip()