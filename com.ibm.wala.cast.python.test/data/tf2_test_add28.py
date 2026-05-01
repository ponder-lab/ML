import tensorflow as tf


def add(a, b):
    return tf.sparse.add(a, b)


arg1 = tf.SparseTensor([[0, 0], [1, 2]], [1, 2], [3, 4])
assert isinstance(arg1, tf.sparse.SparseTensor)
assert arg1.shape == [3, 4]
assert arg1.dtype == tf.int32

arg2 = tf.SparseTensor([[0, 0], [1, 2]], [1, 2], [3, 4])
assert isinstance(arg2, tf.sparse.SparseTensor)
assert arg2.shape == [3, 4]
assert arg2.dtype == tf.int32

c = add(arg1, arg2)
