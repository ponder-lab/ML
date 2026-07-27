# The `np.eye` identity-matrix allocator and its one-hot fancy-indexing idiom (wala/ML#797):
# the allocation types (N, N) float64, and the row-select subscript inherits the dtype.
import numpy as np


def consume(e):
    pass


def consume2(o):
    pass


eye = np.eye(3)
assert eye.shape == (3, 3)
assert eye.dtype == np.float64
consume(eye)

labels = [0, 2, 1, 2]
one_hot = np.eye(3)[labels]
assert one_hot.shape == (4, 3)
assert one_hot.dtype == np.float64
consume2(one_hot)
