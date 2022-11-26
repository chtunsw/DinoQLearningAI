import sys
import logging
import numpy as np
from pathlib import Path
from PIL import Image

file_dir = Path(__file__).parent

logs_dir = file_dir / "logs"
states_dir = file_dir / "states"

# get logger for "train" or "validate" mode
def get_logger(mode):
    # create logger
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # create file handler which logs info level messages
    logs_path = logs_dir / f"{mode}.log"
    fh = logging.FileHandler(logs_path)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger

# save state as image
def save_state_as_image(episode, timestep, state, action, next_state, crashed):
    frames = np.concatenate((state, next_state), axis=1)
    state_filename = f"ep_{episode}_t_{timestep}_action_{action}_crashed_{crashed}.jpg"
    image = Image.fromarray(frames)
    image.save(states_dir / state_filename)
    