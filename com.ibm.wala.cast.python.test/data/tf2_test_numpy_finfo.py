# The float-valued attributes of `np.finfo(dtype)` are NumPy scalars of the queried type
# (wala/ML#907): rank 0, so a tensor scaled by one keeps its shape through the broadcast, and a
# constant folded from one at module level (the subject's `MAX_FLOAT = np.finfo(np.float32).max
# / 100.0`) is scalar too. Every site here queries `np.float32`, so the attributes' dtype is exact;
# `tf2_test_numpy_finfo_dtypes.py` holds the other forms. The integer-valued attributes and
# `np.iinfo` are Python ints and stay unmodeled.
import numpy as np
import tensorflow as tf

MAX_FLOAT = np.finfo(np.float32).max / 100.0


def consume_max32(x):
    pass


def consume_scaled_by_folded_max(x):
    pass


def consume_numpy_scaled(x):
    pass


def consume_iinfo_scaled(x):
    pass


def consume_bits_scaled(x):
    pass


labels = tf.constant(
    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
)  # (3, 4)

# The attribute itself: a float32 scalar.
max32 = np.finfo(np.float32).max
assert max32.shape == () and max32.dtype == np.float32
consume_max32(max32)

# The subject's form: scaled by a module-level constant folded from the attribute.
scaled = labels * MAX_FLOAT
assert scaled.shape == (3, 4) and scaled.dtype == tf.float32
consume_scaled_by_folded_max(scaled)

# A NumPy operand rather than a tensor: the same broadcast.
areas = np.ones((2, 5), dtype=np.float32)
numpy_scaled = areas + np.finfo(np.float32).eps
assert numpy_scaled.shape == (2, 5) and numpy_scaled.dtype == np.float32
consume_numpy_scaled(numpy_scaled)

# Declines (unmodeled, so the product is not typed today): `np.iinfo`, and an integer-valued
# attribute, both Python ints.
iinfo_scaled = labels * np.iinfo(np.int32).max
assert iinfo_scaled.shape == (3, 4)
consume_iinfo_scaled(iinfo_scaled)

bits_scaled = labels * np.finfo(np.float32).bits
assert bits_scaled.shape == (3, 4)
consume_bits_scaled(bits_scaled)
