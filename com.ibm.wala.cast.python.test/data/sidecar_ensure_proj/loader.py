import pickle

import tensorflow as tf


def load_ensured(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return tf.ensure_shape(data, [2, None])
