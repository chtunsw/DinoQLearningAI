import sys
import logging
import re
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

# get top models from train.log
def get_top_models(num):
    logs_path = logs_dir / f"train.log"
    with open(logs_path, "r") as logs_file:
        logs = logs_file.read()
    pattern = re.compile("save model on episode: (.*), average_steps: (.*) \(in recent 10 games\)")
    res = pattern.findall(logs)
    top_records = sorted(res, key=lambda tup: float(tup[1]), reverse=True)[:num]
    top_models = [f"model_weights_{r[0]}.pth" for r in top_records]
    return top_models

# save state as image
def save_state_as_image(episode, timestep, state, action, next_state, crashed):
    frames = np.concatenate((state, next_state), axis=1)
    state_filename = f"ep_{episode}_t_{timestep}_action_{action}_crashed_{crashed}.jpg"
    image = Image.fromarray(frames)
    image.save(states_dir / state_filename)
    