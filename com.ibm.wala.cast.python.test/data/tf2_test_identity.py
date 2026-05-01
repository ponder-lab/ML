import tensorflow as tf


def f(a):
    pass


x = tf.constant([1, 2, 3], tf.int32)
y = tf.identity(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int32

f(y)
