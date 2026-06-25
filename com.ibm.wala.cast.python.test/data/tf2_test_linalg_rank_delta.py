import tensorflow as tf


def consume_diag(a):
    pass


def consume_diag_part(a):
    pass


def consume_matrix_transpose(a):
    pass


def consume_adjoint(a):
    pass


# tf.linalg.diag: (M,) -> (M, M).
d = tf.linalg.diag(tf.constant([1.0, 2.0, 3.0, 4.0]))
assert isinstance(d, tf.Tensor)
assert d.shape == (4, 4)
assert d.dtype == tf.float32
consume_diag(d)

# tf.linalg.diag_part: (M, M) -> (M,).
dp = tf.linalg.diag_part(
    tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
)
assert isinstance(dp, tf.Tensor)
assert dp.shape == (3,)
assert dp.dtype == tf.float32
consume_diag_part(dp)

# tf.linalg.matrix_transpose: (M, N) -> (N, M).
mt = tf.linalg.matrix_transpose(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
assert isinstance(mt, tf.Tensor)
assert mt.shape == (3, 2)
assert mt.dtype == tf.float32
consume_matrix_transpose(mt)

# tf.linalg.adjoint: (M, N) -> (N, M).
adj = tf.linalg.adjoint(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
assert isinstance(adj, tf.Tensor)
assert adj.shape == (3, 2)
assert adj.dtype == tf.float32
consume_adjoint(adj)
