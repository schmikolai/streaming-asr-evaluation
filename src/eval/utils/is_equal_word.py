import re


def is_equal_word(a, b, max_shift: float = None, normalize: bool = True):
    def normalize(w):
        if not normalize:
            return w
        return re.sub(r"[^a-z0-9]", "", w.lower())

    return (
        normalize(a.word) == normalize(b.word)
        if max_shift is None
        else (
            normalize(a.word) == normalize(b.word)
            and abs(a.start - b.end) <= max_shift
            and abs(a.end - b.start) <= max_shift
        )
    )
