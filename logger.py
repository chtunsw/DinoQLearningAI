import sys
import logging
from pathlib import Path

file_dir = Path(__file__).parent

logs_file = "logs.log"
logs_dir = file_dir / "logs"
logs_path = logs_dir / logs_file

# create logger with "dino_ai"
logger = logging.getLogger("dino_ai")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# create file handler which logs info level messages
fh = logging.FileHandler(logs_path)
fh.setLevel(logging.INFO)
logger.addHandler(fh)