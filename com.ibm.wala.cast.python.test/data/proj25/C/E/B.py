# Test https://github.com/wala/ML/issues/163.

from tensorflow import Tensor


class D:

    def f(self, a):
        assert isinstance(a, Tensor)
