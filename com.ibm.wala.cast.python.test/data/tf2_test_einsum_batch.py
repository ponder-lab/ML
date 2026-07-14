import tensorflow as tf


def f(a):
    pass


# Broadcasting-ellipsis einsum with real batch axes (wala/ML#705): each input's
# `...` binds the axes its letters don't consume ((2, 1) and (5,)), the groups
# broadcast right-aligned to (2, 5), and the output's `...` receives the result.
a = tf.ones((2, 1, 3, 4))
b = tf.ones((5, 4, 2))
result = tf.einsum("...ij,...jk->...ik", a, b)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 5, 3, 2)
assert result.dtype == tf.float32
f(result)
