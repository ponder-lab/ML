import tensorflow as tf


def f(a):
    pass


# Diagonal einsum (wala/ML#705): the repeated label within one term names axes
# the runtime requires equal, so its occurrences refine one another and the
# output label composes the single (2,) diagonal dimension.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
result = tf.einsum("ii->i", a)
assert isinstance(result, tf.Tensor)
assert result.shape == (2,)
assert result.dtype == tf.float32
f(result)
