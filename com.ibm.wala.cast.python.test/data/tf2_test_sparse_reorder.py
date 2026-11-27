import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


# `tf.sparse.reorder(sp_input, name=None)`: the result is a `SparseTensor` with the shape and
# dtype of `sp_input` (wala/ML#1351).
x = tf.SparseTensor(indices=[[1, 0], [0, 2]], values=[1.0, 2.0], dense_shape=[2, 3])
y = tf.sparse.reorder(x)
assert isinstance(y, tf.SparseTensor)
assert y.shape == (2, 3)
assert y.dtype == tf.float32
f(y)

# The keyword form.
z = tf.SparseTensor(indices=[[2, 3], [0, 1]], values=[1, 2], dense_shape=[3, 4])
w = tf.sparse.reorder(sp_input=z)
assert isinstance(w, tf.SparseTensor)
assert w.shape == (3, 4)
assert w.dtype == tf.int32
g(w)

# A `SparseTensor` whose `values` the analysis cannot read, here from `sp.find`, keeps its unknown
# shape and dtype through `reorder` rather than being lost (wala/ML#1344).
m = sp.csr_matrix(np.eye(4, dtype=np.float32))
row, col, values = sp.find(m)
u = tf.sparse.reorder(
    tf.SparseTensor(indices=np.array([row, col]).T, values=values, dense_shape=m.shape)
)
assert isinstance(u, tf.SparseTensor)
assert u.shape == (4, 4)
assert u.dtype == tf.float32
h(u)
