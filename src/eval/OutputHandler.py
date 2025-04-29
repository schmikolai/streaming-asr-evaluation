import time
from pydub.utils import re

def norm_word(word) -> str:
    text = word.lower()
    # Remove non-alphabetic characters using regular expression
    text = re.sub(r"[^a-z]", "", text)
    return text.lower().strip().strip(".,?!")


class OutputHandler:
    def __init__(self):
        self.partial_predictions = []
        self.final_words = []

    def init_timer(self, offset: float = 0):
        """
        Initialize the timer for the output handler.
        """
        self.start_time = time.perf_counter() + offset

    def send_partial(self, words):
        """
        Send partial text to the output.
        """
        prediction = {
            "result": words,
            "time": time.perf_counter() - self.start_time
        }
        self.partial_predictions.append(prediction)

    def send_final(self, words, reason: str = None):
        i = len(self.final_words)
        self.final_words += words
        while i < len(self.final_words):
            if (
                norm_word(self.final_words[i]["word"]) == norm_word(self.final_words[i-1]["word"])
                and self.final_words[i]["start"] < self.final_words[i-1]["end"]
            ):
                self.final_words.pop(i)
            else:
                i += 1
