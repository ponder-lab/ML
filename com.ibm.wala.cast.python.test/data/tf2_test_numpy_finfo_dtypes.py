# The dtype half of the `np.finfo` model (wala/ML#907): the attribute's dtype is the queried type's,
# the Python `float` builtin naming float64, and a dtype the program decides at run time between two
# types reading as both. The shape half is rank 0 whatever the dtype. Three sites of two dtypes
# share this file on purpose: the attribute values are produced by one summary helper the analysis
# shares across every `np.finfo` call site, so the dtype read at any one site is the union over all
# the sites in the program, and the union is what the tests pin.
import numpy as np
import tensorflow as tf

EPS_64 = np.finfo(np.float64).eps
TINY_PY = np.finfo(float).tiny


def consume_eps64(x):
    pass


def consume_scaled_by_eps64(x):
    pass


def consume_scaled_by_py_float_tiny(x):
    pass


def consume_scaled_by_two_valued_dtype(x):
    pass


def pick_dtype(flag):
    return np.float32 if flag else np.float64


labels = tf.constant(
    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
)  # (3, 4)

assert EPS_64.shape == () and EPS_64.dtype == np.float64
consume_eps64(EPS_64)

scaled64 = labels * EPS_64
assert scaled64.shape == (3, 4)
consume_scaled_by_eps64(scaled64)

assert TINY_PY.dtype == np.float64
scaled_tiny = labels * TINY_PY
assert scaled_tiny.shape == (3, 4)
consume_scaled_by_py_float_tiny(scaled_tiny)

scaled_two = labels * np.finfo(pick_dtype(len(labels.shape) > 5)).max
assert scaled_two.shape == (3, 4)
consume_scaled_by_two_valued_dtype(scaled_two)
