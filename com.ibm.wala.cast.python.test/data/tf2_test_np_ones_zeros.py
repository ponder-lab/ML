import tensorflow as tf
import numpy as np


def consume_ones(tensor):
    pass


def consume_zeros(tensor):
    pass


def consume_ones_default(tensor):
    pass


# `np.ones(shape, dtype)`: the shape comes from the shape tuple (arg 0) and the
# dtype from the explicit `dtype` argument. Isolated from `tf.constant` so this
# pins the `NpOnes` generator directly (wala/ML#539).
a = np.ones((2, 3), dtype=np.float32)
assert a.shape == (2, 3)
assert a.dtype == np.float32
consume_ones(a)

# `np.zeros(shape, dtype)`: same shape-from-tuple / dtype-from-arg contract via
# `NpZeros`.
b = np.zeros((4,), dtype=np.int64)
assert b.shape == (4,)
assert b.dtype == np.int64
consume_zeros(b)

# `np.ones(shape)` without an explicit dtype: NumPy defaults to `float64` (unlike
# TensorFlow's `tf.ones`, which defaults to `float32`).
c = np.ones((2, 3))
assert c.shape == (2, 3)
assert c.dtype == np.float64
consume_ones_default(c)
