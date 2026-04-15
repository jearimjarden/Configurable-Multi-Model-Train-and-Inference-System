import logging
from datetime import datetime
import json
import traceback
from pathlib import Path
import os
from .schemas import Settings

LOG_FILE_PATH = Path("logs")


class JSONFormatter(logging.Formatter):
    def __init__(self, settings: Settings):
        self.settings = settings

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M"),
            "environment": self.settings.environment,
        }

        if record.exc_info:
            log_record["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            ):
                if value:
                    log_record[key] = value

        return json.dumps(log_record, default=str)


def setup_logging(level: str, settings: Settings) -> None:

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    streamer = logging.StreamHandler()
    streamer.setFormatter(JSONFormatter(settings=settings))
    streamer.setLevel(getattr(logging, level.upper()))

    for handler in root.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root.removeHandler(handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(streamer)

    if settings.save_log:
        os.makedirs(LOG_FILE_PATH, exist_ok=True)
        file_handler = logging.FileHandler(filename=LOG_FILE_PATH / "app.log")
        file_handler.setFormatter(JSONFormatter(settings=settings))
        file_handler.setLevel(getattr(logging, settings.save_log_level.upper()))

        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            root.addHandler(file_handler)


def create_bootstrap_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        level=logging.DEBUG,
    )
    return logging.getLogger("bootstrap")
