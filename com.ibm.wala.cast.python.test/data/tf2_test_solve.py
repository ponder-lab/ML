import tensorflow as tf


def consume_solve(a):
    pass


def consume_cholesky_solve(a):
    pass


def consume_triangular_solve(a):
    pass


# A (3, 3) coefficient matrix and a (3, 5) right-hand side. The solve-family
# results all share the right-hand side's (3, 5) shape and float32 dtype, not
# the coefficient matrix's (3, 3) shape. See wala/ML#513.
matrix = tf.constant([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
rhs = tf.constant(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
    ]
)

s = tf.linalg.solve(matrix, rhs)
assert isinstance(s, tf.Tensor)
assert s.shape == (3, 5)
assert s.dtype == tf.float32
consume_solve(s)

cs = tf.linalg.cholesky_solve(matrix, rhs)
assert isinstance(cs, tf.Tensor)
assert cs.shape == (3, 5)
assert cs.dtype == tf.float32
consume_cholesky_solve(cs)

ts = tf.linalg.triangular_solve(matrix, rhs)
assert isinstance(ts, tf.Tensor)
assert ts.shape == (3, 5)
assert ts.dtype == tf.float32
consume_triangular_solve(ts)
