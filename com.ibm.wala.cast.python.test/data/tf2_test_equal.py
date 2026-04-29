import tensorflow as tf


def f(a):
    pass


x = tf.constant([[1.0, 1.0], [2.0, 2.0]])
y = tf.constant([[1.0, 2.0], [2.0, 1.0]])

z = tf.equal(x, y)
assert isinstance(z, tf.Tensor)
assert z.dtype == tf.bool
assert z.shape == (2, 2)

f(z)
