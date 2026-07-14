import tensorflow as tf


def f(a):
    pass


# Broadcasting-ellipsis einsum (wala/ML#705): each `...` binds the axes its
# letters don't consume (here none, so the groups are empty) and the output's
# `...` receives the broadcast result, composing the same (2, 2) shape as
# "ij,jk->ik".
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
result = tf.einsum("...ij,...jk->...ik", a, b)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 2)
assert result.dtype == tf.float32
f(result)
