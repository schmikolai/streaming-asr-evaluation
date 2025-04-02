from src.BaseTranscriberAdapter import BaseTranscriberAdapter
import websockets
import logging
logger = logging.getLogger(__name__)

class WebsocketTranscriberAdapter(BaseTranscriberAdapter):
    def __init__(self, websocket_url):
        self.websocket_url = websocket_url

    def _start_stream(self) -> bool:
        self.websocket = websockets.connect(self.websocket_url)
        return True
    
    def _stop_stream(self) -> bool:
        self.websocket.close()
        return True

    def transcribe(self, audio_bytes) -> str:
        self.websocket.send(audio_bytes)