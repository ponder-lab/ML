import tensorflow as tf


def f(a):
    pass


tensor = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
mask = tf.constant([True, False, True])
y = tf.boolean_mask(tensor, mask)
assert isinstance(y, tf.Tensor)
assert y.shape == (2, 2)
assert y.dtype == tf.float32

f(y)
