# The numpy dtype-token witnesses of wala/ML#775: the `np.ndarray` constructor with an
# `np.int32` token, `np.array` with the Python builtin `int` (which numpy promotes to int64),
# and an allocator with the `np.uint8` token.
import numpy as np


def consume(a):
    pass


def consume2(b):
    pass


def consume3(c):
    pass


a = np.ndarray(shape=(3,), dtype=np.int32)
assert a.dtype == np.int32
assert a.shape == (3,)
consume(a)

b = np.array([[1, 2], [3, 4]], dtype=int)
assert b.dtype == np.int64
assert b.shape == (2, 2)
consume2(b)

c = np.zeros((2,), dtype=np.uint8)
assert c.dtype == np.uint8
assert c.shape == (2,)
consume3(c)
