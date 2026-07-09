import tensorflow as tf


def f(a):
    pass


# Implicit-output einsum (no `->`): the output is the labels occurring exactly
# once, in alphabetical order, so `"ij,jk"` is equivalent to `"ij,jk->ik"`.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
result = tf.einsum("ij,jk", a, b)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 2)
assert result.dtype == tf.float32
f(result)
