# Test https://github.com/wala/ML/issues/1343: the dtype of `tf.sparse.add` of a sparse tensor
# whose `values` the analysis cannot read, here a result of the unmodeled `tf.sparse.reduce_sum`,
# is read as unknown instead of failing the analysis, and then filled from the other operand.
import tensorflow as tf


def f(a):
    pass


st = tf.sparse.SparseTensor(
    indices=[[0, 0], [1, 1]], values=[1.0, 2.0], dense_shape=[2, 2]
)
d = tf.sparse.SparseTensor(
    indices=[[0, 0], [1, 1]],
    values=tf.sparse.reduce_sum(st, axis=-1),
    dense_shape=[2, 2],
)
c = tf.sparse.add(d, st)
assert c.dtype == tf.float32
assert c.shape == (2, 2)
f(c)
