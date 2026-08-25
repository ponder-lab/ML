# wala/ML#835: `np.transpose` and `ndarray.transpose` were unmodeled, so a transposed array was
# untyped and everything downstream degraded.
import numpy as np


def consume_permuted(x):
    assert x.shape == (4, 2, 3)
    assert x.dtype == np.float32


def consume_reversed(x):
    assert x.shape == (4, 3, 2)
    assert x.dtype == np.float32


def consume_method(x):
    assert x.shape == (3, 2)
    assert x.dtype == np.float32


a = np.zeros((2, 3, 4), dtype=np.float32)
consume_permuted(np.transpose(a, (2, 0, 1)))
consume_reversed(np.transpose(a))

b = np.ones((2, 3), dtype=np.float32)
consume_method(b.transpose())
