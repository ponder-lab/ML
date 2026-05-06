import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.broadcast_to(x, [2, 3])
assert isinstance(y, tf.Tensor)
assert y.shape == (2, 3)
assert y.dtype == tf.float32

f(y)
