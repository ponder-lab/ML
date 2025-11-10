import tensorflow as tf


def add(a, b):
    return a + b


a = tf.random.gamma([10], [0.5, 1.5])
assert isinstance(a, tf.Tensor)
assert a.shape == (10, 2)
assert a.dtype == tf.float32

b = tf.random.gamma([10], [1, 2.5])
assert isinstance(a, tf.Tensor)
assert b.shape == (10, 2)
assert a.dtype == tf.float32

c = add(a, b)
