import tensorflow as tf


def f(a):
    pass


y = tf.sequence_mask([1, 3, 2], maxlen=5)
assert isinstance(y, tf.Tensor)
assert y.shape == (3, 5)
assert y.dtype == tf.bool

f(y)
