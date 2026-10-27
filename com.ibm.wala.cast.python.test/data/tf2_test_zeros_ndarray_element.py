# An element of an ndarray in an `np.zeros` shape tuple makes the whole shape unknown
# (wala/ML#1158).
import numpy as np


def consume(x):
    pass


sizes = 416 // np.array([8, 16, 32])
label = np.zeros((4, sizes[0], sizes[0], 3, 7), dtype=np.float32)
assert label.shape == (4, 52, 52, 3, 7)
assert label.dtype == np.float32
consume(label)


def consume_mixed(x):
    pass


def pick(flag):
    return sizes[1] if flag else 26


# A position that may be an ndarray element or a literal keeps both.
for flag in (True, False):
    mixed = np.zeros((4, pick(flag)), dtype=np.float32)
    assert mixed.shape == (4, 26)
    consume_mixed(mixed)
