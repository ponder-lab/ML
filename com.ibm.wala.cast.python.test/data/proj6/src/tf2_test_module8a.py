# Test https://github.com/wala/ML/issues/163.

from tensorflow import Tensor


class C:

    def f(self, a):
        assert isinstance(a, Tensor)


class D:

    def g(self, a):
        assert isinstance(a, Tensor)
