# Test of tf.math.reciprocal as a pure passthrough of shape and dtype.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 4.0])
y = tf.math.reciprocal(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
