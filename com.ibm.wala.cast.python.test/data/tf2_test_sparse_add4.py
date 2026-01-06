import tensorflow as tf


def f(a):
    pass


t1 = ((1, 0), (0, 2))

st2 = tf.sparse.SparseTensor(
    indices=[[0, 1], [1, 0]], values=[3, 4], dense_shape=[2, 2]
)

c = tf.sparse.add(st2, t1)
assert c.shape == (2, 2)
assert c.dtype == tf.int32

f(c)
