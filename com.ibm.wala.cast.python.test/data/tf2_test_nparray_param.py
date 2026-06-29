import numpy as np


def f(t):
    return t


# `x` is a bare `numpy.array` value. See wala/ML#598.
x = np.array([1.0, 2.0, 3.0])
assert x.shape == (3,)
# Runtime dtype is float64 (numpy promotes the Python floats); the static type is unknown
# because numpy dtype promotion is unmodeled. See wala/ML#626.
assert x.dtype == np.float64
f(x)
