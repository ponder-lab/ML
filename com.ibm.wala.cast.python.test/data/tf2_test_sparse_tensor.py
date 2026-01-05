import tensorflow as tf


def f(x):
    return x


s = tf.sparse.SparseTensor([[0, 0], [1, 2]], [1, 2], [3, 4])
assert isinstance(s, tf.sparse.SparseTensor)
assert s.dtype == tf.int32
assert s.shape == (3, 4)

f(s)
