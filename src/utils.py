import logging
from datetime import datetime
import os
import sys


class DynamicPathFileHandler(logging.FileHandler):
    def __init__(
        self,
        directory,
        filename,
        mode="a",
        encoding=None,
        delay=False,
    ):
        self.base_directory = directory
        self.base_filename = filename
        filepath = self._calculate_dynamic_path()

        super().__init__(filepath, mode, encoding, delay)

    def _calculate_dynamic_path(self):
        date_now = datetime.now()
        directory = os.path.join(
            self.base_directory, str(date_now.year), date_now.strftime("%B")
        )
        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(
            directory, date_now.strftime("%d%m%Y") + self.base_filename
        )
        self.currently_logging_to = filepath
        return filepath

    def emit(self, record):
        module_name = os.path.basename(
            sys.argv[0] if sys.argv[0] else "unknown_program"
        )
        record_program = os.path.splitext(module_name)[0]
        super().emit(record)


def setup_logger(name=__name__):
    LOG_DIR = os.environ.get("LOG_DIR")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = DynamicPathFileHandler(directory=LOG_DIR, filename=".log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
