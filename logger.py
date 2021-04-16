import os


class Logger:
    def __init__(self, file_name: str = "log.txt", output_dir: str = None):
        if output_dir is None:
            self.file_path = f"{file_name}.txt"
        else:
            self.file_path = os.path.join(output_dir, f"{file_name}.txt")
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
