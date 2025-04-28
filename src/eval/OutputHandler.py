class OutputHandler:
    def __init__(self):
        self.messages = []

    def send_text(self, text):
        """
        Send text to the output.
        """
        self.messages.append(text)