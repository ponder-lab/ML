from tensorflow import Tensor


def g(a):
    assert isinstance(a, Tensor)
    assert a.shape == (1, 2)
