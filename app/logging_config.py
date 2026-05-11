import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for existing_handler in list(root_logger.handlers):
        root_logger.removeHandler(existing_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        rename_fields={
            "asctime": "timestamp", 
            "levelname": "level", 
            "name": "logger_name" 
        },
    )
    
    stream_handler.setFormatter(json_formatter)
    root_logger.addHandler(stream_handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    logging.info("Logging system initialized with JSON formatter.")