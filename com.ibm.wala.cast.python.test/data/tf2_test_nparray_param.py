import numpy as np


def f(t):
    # `t` is a bare `numpy.array` value. See wala/ML#598.
    assert t.shape == (3,)
    # Runtime dtype is float64 (numpy promotes the Python floats); the static type is unknown
    # because numpy dtype promotion is unmodeled. See wala/ML#626.
    assert t.dtype == np.float64
    return t


x = np.array([1.0, 2.0, 3.0])
f(x)
