import tensorflow as tf


def f(a):
    pass


x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.math.argmax(x, axis=0)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int64

f(y)
