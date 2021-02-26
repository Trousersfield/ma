import os

script_dir = os.path.abspath(os.path.dirname(__file__))


class Logger:
    def __init__(self):
        self.file_path = "log.txt"
        file = open(self.file_path, "w+")
        file.close()

    def clear(self):
        file = open(self.file_path, "w")
        file.flush()

    def write(self, content: str) -> None:
        if len(content) == 0:
            return
        file = open(self.file_path, "a")
        file.write(content + "\n")
        file.close()
