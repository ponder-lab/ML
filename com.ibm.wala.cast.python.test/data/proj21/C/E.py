# Test https://github.com/wala/ML/issues/163.

from tensorflow import Tensor


class G:

    def g(self, a):
        assert isinstance(a, Tensor)