import tensorflow as tf


def consume(a):
    pass


# `tf.tensor_scatter_nd_update` returns a copy of `tensor` with the rows
# at `indices` replaced, so the result has the shape and dtype of `tensor`
# alone: a rank-3 float32 tensor of zeros here.
tensor = tf.zeros([4, 3, 4], dtype=tf.float32)
indices = tf.constant([[0], [2]])
updates = tf.ones([2, 3, 4], dtype=tf.float32)
y = tf.tensor_scatter_nd_update(tensor, indices, updates)
assert isinstance(y, tf.Tensor)
assert y.shape == (4, 3, 4)
assert y.dtype == tf.float32

consume(y)
