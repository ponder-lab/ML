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
