import tensorflow as tf


class MyClass:

    @staticmethod
    def the_static_method(x):
        assert isinstance(x, tf.Tensor)


a = tf.constant(1)

assert a.shape == ()  # scalar
assert a.dtype == tf.int32

MyClass.the_static_method(a)
