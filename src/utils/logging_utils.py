import logging
from pathlib import Path
from datetime import datetime
import time

def create_logger(log_dir: Path, script_name: str):
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filename timestamp in local time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Handlers
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Force logging timestamps to local timezone
    logging.Formatter.converter = time.localtime   # <---- THIS FIXES IT

    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized for {script_name}")

    return logger, log_file