import numpy as np


def f(t):
    return t


# numpy infers complex128 for a complex literal array. See wala/ML#626.
x = np.array([1j, 2j])
assert x.shape == (2,)
assert x.dtype == np.complex128
f(x)
