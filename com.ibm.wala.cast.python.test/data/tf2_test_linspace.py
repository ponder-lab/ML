import tensorflow as tf


def f(a):
    pass


y = tf.linspace(0.0, 10.0, 5)
assert isinstance(y, tf.Tensor)
assert y.shape == (5,)
assert y.dtype == tf.float32

f(y)
