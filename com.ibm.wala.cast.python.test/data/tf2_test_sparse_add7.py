import tensorflow as tf


def test(t1, t2):
    pass


# Use SparseTensor literals instead of from_dense to ensure points-to sets are non-empty in WALA
a = tf.sparse.SparseTensor(indices=[[0, 0], [1, 1]], values=[1, 2], dense_shape=[2, 2])
b = tf.sparse.SparseTensor(indices=[[0, 1], [1, 0]], values=[1, 3], dense_shape=[2, 2])

# Positional (a, b) and keyword (threshold)
t1 = tf.sparse.add(a, b, threshold=0)
assert t1.shape == (2, 2)
assert t1.dtype == tf.int32

# Keyword args
t2 = tf.sparse.add(a=a, b=b)
assert t2.shape == (2, 2)
assert t2.dtype == tf.int32

test(t1, t2)
