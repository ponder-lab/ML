# Test https://github.com/wala/ML/issues/1047: class methods called through their class, under
# names other than the one the original fixture used.
import tensorflow as tf


def consume(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == ()
    assert x.dtype == tf.int32


class MyClass:

    def f(x):
        assert isinstance(x, tf.Tensor)

    def __init__(self, x):
        self.x = MyClass.cast(x)

    @classmethod
    def from_value(cls, x):
        assert isinstance(x, tf.Tensor)
        cls.f(x)

    @classmethod
    def cast(cls, x):
        return x


MyClass.from_value(tf.constant(1))
consume(MyClass(tf.constant(2)).x)
