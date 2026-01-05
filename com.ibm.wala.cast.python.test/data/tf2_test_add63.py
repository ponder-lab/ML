import tensorflow as tf


def add(a, b):
    return tf.sparse.add(a, b)


arg = tf.sparse.SparseTensor([[0, 0], [1, 2]], [1, 2], [3, 4])
assert isinstance(arg, tf.sparse.SparseTensor)
assert arg.dtype == tf.int32
assert arg.shape == (3, 4)

c = add(
    arg,
    arg,
)
