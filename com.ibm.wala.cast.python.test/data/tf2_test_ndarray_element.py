# The element of an array: binding a row by iterating a 2-D array, by a constant index, or by a
# loop-carried index. Each is an array of the receiver's dtype one rank down, and the arithmetic
# over its slices promotes as NumPy does, so a float literal beside the integral row reads
# float64, not the float32 that an unresolved operand falls to.
import numpy as np


def consume_loop(t):
    assert t.shape == (4,), t.shape
    assert t.dtype == np.float64, t.dtype


def consume_index(t):
    assert t.shape == (4,), t.shape
    assert t.dtype == np.float64, t.dtype


def consume_loop_index(t):
    assert t.shape == (4,), t.shape
    assert t.dtype == np.float64, t.dtype


def consume_row(t):
    assert t.shape == (5,), t.shape
    assert t.dtype == np.int64, t.dtype


def xywh(bc):
    return np.concatenate([(bc[2:] + bc[:2]) * 0.5, bc[2:] - bc[:2]], axis=-1)


bbs = np.array([[10, 20, 30, 40, 1], [5, 6, 7, 8, 2]])

for bb in bbs:
    consume_row(bb)
    consume_loop(1.0 * xywh(bb[:4]))

bi = bbs[0]
consume_index(1.0 * xywh(bi[:4]))

for i in range(2):
    consume_loop_index(1.0 * xywh(bbs[i][:4]))
