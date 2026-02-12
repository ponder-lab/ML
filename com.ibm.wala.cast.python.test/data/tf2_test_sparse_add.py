import tensorflow as tf


def f(a):
    pass


# Shape (2, 2)
st1 = tf.sparse.SparseTensor(
    indices=[[0, 0], [1, 1]], values=[1.0, 2.0], dense_shape=[2, 2]
)
st2 = tf.sparse.SparseTensor(
    indices=[[0, 1], [1, 0]], values=[3.0, 4.0], dense_shape=[2, 2]
)

c = tf.sparse.add(st1, st2)
assert c.shape == (2, 2)
assert c.dtype == tf.float32

f(c)
