# Test of tf.math.round as a pure passthrough of shape and dtype.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([0.4, 0.6, 1.4])
y = tf.math.round(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
