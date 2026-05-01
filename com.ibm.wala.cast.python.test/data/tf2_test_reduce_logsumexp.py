import tensorflow as tf


def f(a):
    pass


x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.math.reduce_logsumexp(x)
assert isinstance(y, tf.Tensor)
assert y.shape == ()
assert y.dtype == tf.float32

f(y)
