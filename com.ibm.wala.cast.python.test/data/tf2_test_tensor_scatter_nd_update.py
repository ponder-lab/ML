import tensorflow as tf


def f(a):
    pass


tensor = tf.constant([1.0, 2.0, 3.0, 4.0])
indices = tf.constant([[0], [2]])
updates = tf.constant([5.0, 6.0])
y = tf.tensor_scatter_nd_update(tensor, indices, updates)
assert isinstance(y, tf.Tensor)
assert y.shape == (4,)
assert y.dtype == tf.float32

f(y)
