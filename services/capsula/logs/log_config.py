from pydantic import BaseModel

class LogConfig(BaseModel):
    LOGGER_NAME:str = "CAPSUL-IA"
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "INFO"
    LOG_FILE_TARGET: str = "./logs/capsule.log"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    }
    handlers: dict = {
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "level": "INFO",
            "filename": LOG_FILE_TARGET,
            "mode": "a",
            "encoding": "utf-8",
            "maxBytes": 1000000,
            "backupCount": 10,
        }
    }
    loggers: dict = {
        LOGGER_NAME: {"handlers": ["console", "file"], "level": LOG_LEVEL},
    }