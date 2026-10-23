import pickle

import numpy as np


def load_mean(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.mean(data, dtype=np.dtype("float32"))
