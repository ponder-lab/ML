from tensorflow import Tensor


def f(a):
    assert isinstance(a, Tensor)
    assert a.shape == (1, 2)
