import numpy as np


def f(t):
    return t


# A nested literal mixing ints and a float: numpy promotes to `float64`, and the static walk
# descends through the nested lists to find the float leaf. See wala/ML#626.
x = np.array([[1, 2.0], [3, 4]])
assert x.shape == (2, 2)
assert x.dtype == np.float64
f(x)
