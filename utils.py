import sys
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
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

# get train history from train.log
def get_train_history():
    logs_path = logs_dir / f"train.log"
    with open(logs_path, "r") as logs_file:
        logs = logs_file.read()
    pattern = re.compile("save model on episode: (.*), average_steps: (.*) \(in recent 10 games\)")
    train_history = pattern.findall(logs)
    return train_history

# plot train history
def plot_train_history():
    train_history = get_train_history()
    episodes = [int(e[0]) for e in train_history]
    steps = [float(e[1]) for e in train_history]
    plt.plot(episodes, steps)
    plt.title("training history")
    plt.xlabel("episodes")
    plt.ylabel("average steps (per 10 games)")
    plt.xticks(np.arange(min(episodes), max(episodes), 1000))
    plt.yticks(np.arange(min(steps), max(steps), 10))
    plt.show()

# get top models from train history
def get_top_models(num):
    train_history = get_train_history()
    top_records = sorted(train_history, key=lambda tup: float(tup[1]), reverse=True)[:num]
    top_models = [f"model_weights_{r[0]}.pth" for r in top_records]
    return top_models

# save state as image
def save_state_as_image(episode, timestep, state, action, next_state, crashed):
    frames = np.concatenate((state, next_state), axis=1)
    state_filename = f"ep_{episode}_t_{timestep}_action_{action}_crashed_{crashed}.jpg"
    image = Image.fromarray(frames)
    image.save(states_dir / state_filename)
