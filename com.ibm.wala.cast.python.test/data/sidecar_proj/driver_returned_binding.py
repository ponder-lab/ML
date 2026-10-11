# Fixture for wala/ML#1046: an annotation on a binding that its function returns.
#
# The entry fills `make`'s `x`. The caller's `out` holds the same array, but its own seed is read
# from the `np.zeros` that created it, so before the fix the caller saw the unfilled type beside
# the filled one.
import numpy as np


def consume(x):
    pass


data = [np.array([0, 3, 1]), np.array([2, 0])]
depth = np.max([np.max(labels) for labels in data]) + 1


def make(labels):
    num = len(labels)
    x = np.zeros([num, depth], dtype=np.float32)
    x[list(range(num)), labels] = 1.0
    return x


for labels in data:
    out = make(labels)
    consume(out)
    assert out.shape == (len(labels), 4)
    assert out.dtype == np.float32
