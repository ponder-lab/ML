# Test https://github.com/wala/ML/issues/163.

from tensorflow import Tensor


def g(a):
    assert isinstance(a, Tensor)
