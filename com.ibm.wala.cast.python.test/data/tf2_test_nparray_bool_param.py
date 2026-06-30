import numpy as np


def f(t):
    return t


# numpy infers `bool` for an all-boolean literal array. The static analysis models this. See
# wala/ML#626.
x = np.array([True, False])
assert x.shape == (2,)
assert x.dtype == np.bool_
f(x)
