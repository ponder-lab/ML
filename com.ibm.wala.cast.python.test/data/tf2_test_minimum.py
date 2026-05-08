# Test of tf.math.minimum as an elementwise binary op.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y_in = tf.constant([3.0, 2.0, 1.0])
y = tf.math.minimum(x, y_in)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
