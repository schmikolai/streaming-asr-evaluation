import time

class OutputHandler:
    def __init__(self):
        self.messages = []

    def init_timer(self, offset: float = 0):
        """
        Initialize the timer for the output handler.
        """
        self.start_time = time.perf_counter() + offset

    def send_partial(self, text):
        """
        Send partial text to the output.
        """
        message = {
            "partial": text,
            "time": time.perf_counter() - self.start_time
        }
        self.messages.append(message)

    def send_final(self, text, reason: str = None):
        message = {
            "final": text,
            "reason": reason,
            "time": time.perf_counter() - self.start_time
        }
        self.messages.append(message)
