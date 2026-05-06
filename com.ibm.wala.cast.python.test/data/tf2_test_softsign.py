# Test of tf.math.softsign as a pure passthrough of shape and dtype.
import tensorflow as tf


def f(a):
    pass


features = tf.constant([-1.0, 0.0, 1.0])
y = tf.math.softsign(features)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
