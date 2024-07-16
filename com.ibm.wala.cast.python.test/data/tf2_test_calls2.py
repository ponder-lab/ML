import tensorflow as tf
from tf2_test_calls3 import g


class C:

    def f(self, a):
        assert isinstance(a, tf.Tensor)


g(C())
