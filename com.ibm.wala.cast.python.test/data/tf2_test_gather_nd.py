import tensorflow as tf


def f(a):
    pass


params = tf.constant([[1.0, 2.0], [3.0, 4.0]])
indices = tf.constant([[0, 0], [1, 1]])
y = tf.gather_nd(params, indices)
assert isinstance(y, tf.Tensor)
assert y.shape == (2,)
assert y.dtype == tf.float32

f(y)
