import tensorflow as tf


def f(a):
    pass


# Keyword-argument variant of `tf2_test_einsum.py`. Einsum's API is
# `einsum(equation, *inputs, **kwargs)`; with a variadic positional `*inputs`,
# the kwargs you can name explicitly are `name` and `optimize`. This fixture
# uses both. Same dtype/shape expectation as the positional variant.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
result = tf.einsum("ij,jk->ik", a, b, name="kw_test", optimize="optimal")
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 2)
assert result.dtype == tf.float32
f(result)
