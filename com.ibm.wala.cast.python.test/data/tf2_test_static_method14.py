import tensorflow as tf


class MyClass:

    @staticmethod
    def the_static_method(x):
        assert isinstance(x, tf.Tensor)


a = tf.constant(1, tf.float32, ([1, 2]))

assert a.shape == (1, 2)
assert a.dtype == tf.float32

MyClass.the_static_method(a)
