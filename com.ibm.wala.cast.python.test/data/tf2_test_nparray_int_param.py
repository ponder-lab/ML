import numpy as np


def f(t):
    return t


# `x` is a bare `numpy.array` of Python ints. numpy promotes Python `int` to `int64` (not the
# `int32` TF-literal convention), and the static analysis models this promotion. See wala/ML#626.
x = np.array([1, 2, 3])
assert x.shape == (3,)
assert x.dtype == np.int64
f(x)
