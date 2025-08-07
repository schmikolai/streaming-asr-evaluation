def word_sequence_to_string(word_sequence) -> str:
    return " ".join([w.word for w in word_sequence])

def word_dict_sequence_to_string(word_sequence) -> str:
    return " ".join([w['word'] for w in word_sequence])