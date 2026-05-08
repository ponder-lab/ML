import tensorflow as tf


def f(a):
    pass


# `tf.einsum` is the dispatch test. Generator emits ⊤ shape (since the
# precise shape requires equation parsing) but inherits the dtype from
# the first tensor input (`a` here, which is float32).
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
result = tf.einsum("ij,jk->ik", a, b)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 2)
assert result.dtype == tf.float32
f(result)
