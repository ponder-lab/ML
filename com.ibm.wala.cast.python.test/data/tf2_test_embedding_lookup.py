import tensorflow as tf


def f(a):
    pass


params = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
ids = tf.constant([0, 2])
y = tf.nn.embedding_lookup(params, ids)
assert isinstance(y, tf.Tensor)
assert y.shape == (2, 2)
assert y.dtype == tf.float32

f(y)
