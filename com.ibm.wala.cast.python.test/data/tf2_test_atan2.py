# Test of tf.math.atan2 as an elementwise binary op.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([0.0, 0.5, 1.0])
y_in = tf.constant([1.0, 1.0, 1.0])
y = tf.math.atan2(x, y_in)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
