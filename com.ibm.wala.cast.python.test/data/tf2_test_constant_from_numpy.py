# Minimal fixture isolating the `numpy → tf.constant` dtype-loss bug
# (wala/ML#539). Ariadne loses dtype info when a tensor is constructed via
# `tf.constant(np_array)` rather than from a literal Python list/scalar or a
# modeled API (mnist load_data, Dense layer chain, etc.).
import numpy as np
import tensorflow as tf


def consume(x):
    """Sink function — pins the dtype/shape of whatever flows in as `x`."""
    return x


# Path A: `tf.constant` of a numpy array. The hypothesis says dtype is lost here.
arr_from_np = tf.constant(np.ones((2, 3), dtype=np.float32))
assert arr_from_np.shape == (2, 3)
assert arr_from_np.dtype == tf.float32
consume(arr_from_np)
