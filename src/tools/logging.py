import logging
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.now().strftime("%d/%m/%Y, %H:%M"),
        }

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

        return json.dumps(log_record)


def logging_setup(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    streamer = logging.StreamHandler()
    streamer.setFormatter(JSONFormatter())
    streamer.setLevel(getattr(logging, level.upper()))

    if not any(isinstance(handle, logging.StreamHandler) for handle in root.handlers):
        root.addHandler(streamer)
