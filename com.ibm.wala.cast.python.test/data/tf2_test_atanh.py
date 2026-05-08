# Test of tf.math.atanh as a pure passthrough of shape and dtype.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([0.0, 0.3, 0.5])
y = tf.math.atanh(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
