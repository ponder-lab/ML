# A subscript adding an axis to an array a Keras dataset loader returns is a new NumPy array, as
# one of an `ndarray` is (https://github.com/wala/ML/issues/400).
import sys

import numpy as np
import tensorflow as tf


def consume_ellipsis_none(x):
    pass


def consume_slice(x):
    pass


def consume_scaled(x):
    pass


def consume_slice_ellipsis_none(x):
    pass


def consume_unknown_dtype(x):
    pass


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x1 = x_train[..., None]
assert x1.shape == (60000, 28, 28, 1) and x1.dtype == np.uint8
consume_ellipsis_none(x1)

x2 = x_train[:2]
assert x2.shape == (2, 28, 28) and x2.dtype == np.uint8
consume_slice(x2)

x3 = x_train[..., None].astype(np.float32) / 255.0
assert x3.shape == (60000, 28, 28, 1) and x3.dtype == np.float32
consume_scaled(x3)

x4 = x_train[:2, ..., None]
assert x4.shape == (2, 28, 28, 1) and x4.dtype == np.uint8
consume_slice_ellipsis_none(x4)

# A dtype named by a string the analysis does not fold.
pair = (np.ones((3, 4), dtype=np.dtype(sys.argv[0][:0] + "float32")), 0)
unknown, _ = pair
x5 = unknown[..., None]
assert x5.shape == (3, 4, 1) and x5.dtype == np.float32
consume_unknown_dtype(x5)
