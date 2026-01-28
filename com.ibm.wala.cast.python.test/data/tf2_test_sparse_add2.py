import tensorflow as tf


def f(a):
    pass


t1 = ((1.0, 0.0), (0.0, 2.0))

st2 = tf.sparse.SparseTensor(
    indices=[[0, 1], [1, 0]], values=[3.0, 4.0], dense_shape=[2, 2]
)

c = tf.sparse.add(t1, st2)
assert c.shape == (2, 2)
assert c.dtype == tf.float32

f(c)
