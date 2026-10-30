from tensorflow import Tensor


def h(a):
    assert isinstance(a, Tensor)
    assert a.shape == (1, 2)
