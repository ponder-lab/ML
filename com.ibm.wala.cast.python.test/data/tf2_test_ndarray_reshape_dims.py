# An array's `reshape` given its dimensions as separate integers, `x.reshape(2, 3)`, as well as one
# tuple or one integer. NumPy reads every positional integer as a dimension.
import numpy as np


def consume_two(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_three(t):
    assert t.shape == (1, 2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_row(t):
    assert t.shape == (3,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_subscript(t):
    assert t.shape == (3,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_inferred(t):
    assert t.shape == (3, 2) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_one(t):
    assert t.shape == (6,) and t.dtype == np.int64, (t.shape, t.dtype)


def consume_tuple(t):
    assert t.shape == (2, 3) and t.dtype == np.int64, (t.shape, t.dtype)


x = np.arange(6).reshape(2, 3)
consume_two(x)
consume_three(np.arange(6).reshape(1, 2, 3))
for row in x:
    consume_row(row)
consume_subscript(x[0])
consume_inferred(np.arange(6).reshape(-1, 2))
consume_one(np.arange(6).reshape(6))
consume_tuple(np.arange(6).reshape((2, 3)))
