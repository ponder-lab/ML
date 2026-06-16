import tensorflow as tf


def f(a):
    pass


x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.linalg.trace(x)
assert isinstance(y, tf.Tensor)
# The trace collapses the last two dimensions, so a (2, 2) input yields a scalar.
assert y.shape == ()
assert y.dtype == tf.float32

f(y)
