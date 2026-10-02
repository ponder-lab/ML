# https://github.com/wala/ML/issues/1009: a slice of an array, and the result of arithmetic on
# one, keep the array's methods.
import numpy as np


def consume_astype(x):
    return x


def consume_tolist(x):
    return x


def consume_scaled(x):
    return x


x = np.ones((4, 3), dtype=np.float32)
y = x[1:3]
cast = y.astype(np.int32)
assert cast.shape == (2, 3) and cast.dtype == np.int32
consume_astype(cast)
consume_tolist(np.array(y.tolist()))
scaled = (x * 2.0).astype(np.int32)
assert scaled.shape == (4, 3) and scaled.dtype == np.int32
consume_scaled(scaled)
