import pickle

import numpy as np


def load_clip(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.clip(data, 0, 5)


def load_clip_by_negative(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.clip(data, -1, 1)


def load_clip_by_array(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.clip(data, np.zeros((3, 1, 1), dtype=np.float64), 5)


def load_expanded(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.expand_dims(data, 0)


def load_squeezed(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.squeeze(data)
