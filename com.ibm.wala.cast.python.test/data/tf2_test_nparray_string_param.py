import numpy as np


def f(t):
    return t


# numpy infers a string dtype for an all-string literal array. The static analysis models this. See
# wala/ML#626.
x = np.array(["a", "b"])
assert x.shape == (2,)
assert x.dtype.kind == "U"
f(x)
