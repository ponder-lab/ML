# Test an `astype` whose dtype is another array's `.dtype`: the result takes that array's dtype.
import numpy as np


def consume_positional(a):
    assert a.shape == (2, 3) and a.dtype == np.float32


def consume_keyword(a):
    assert a.shape == (2, 3) and a.dtype == np.float32


def consume_in_method(a):
    assert a.shape == (2, 3) and a.dtype == np.float32


def consume_literal(a):
    assert a.shape == (2, 3) and a.dtype == np.float32


class Holder:
    def __init__(self, x, w):
        self.x = x
        self.w = w

    def matched(self, x_c, w_c):
        w_c = w_c.astype(x_c.dtype)
        consume_in_method(w_c)


x = np.ones((2, 3), np.float32)
w = np.zeros((2, 3), np.int64)

consume_positional(w.astype(x.dtype))
consume_keyword(w.astype(dtype=x.dtype))
consume_literal(w.astype(np.float32))

holder = Holder(x, w)
holder.matched(holder.x, holder.w)
