import tensorflow as tf


def f(a):
    pass


# `tf.sparse.from_dense(tensor, name=None)`: shape inherits from `tensor`;
# dtype inherits from `tensor`. Result is a `SparseTensor` (not a regular
# `Tensor`). The keyword form below ensures the analyzer's arg-resolution
# uses `SparseFromDense.Parameters.TENSOR.getName()` to find the input,
# closing the wala/ML#510 coverage gap on `Parameters#getName()`.
x = tf.constant([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
assert isinstance(x, tf.Tensor)
assert x.shape == (2, 3)
assert x.dtype == tf.float32
y = tf.sparse.from_dense(tensor=x)
assert isinstance(y, tf.SparseTensor)
assert y.shape == (2, 3)
assert y.dtype == tf.float32
f(y)
