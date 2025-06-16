from assemblyai.types import Word
from assemblyai.streaming.v3 import TurnEvent
from typing import List

def serialize_word(w: Word):
    return {
        "word": w.text,
        "conf": w.confidence,
        "start": float(w.start) / 1000,
        "end": float(w.end) / 1000
    }

def result_from_turn(turn: TurnEvent):
    return {
        "text": turn.transcript,
        "result": list(map(serialize_word, turn.words))
    }