import tensorflow as tf


def consume(a):
    pass


# An integer `tensor` gives an integer result of the same shape.
tensor = tf.constant([1, 2, 3, 4, 5, 6])
indices = tf.constant([[1], [3]])
updates = tf.constant([9, 10])
y = tf.tensor_scatter_nd_update(tensor, indices, updates)
assert isinstance(y, tf.Tensor)
assert y.shape == (6,)
assert y.dtype == tf.int32

consume(y)
