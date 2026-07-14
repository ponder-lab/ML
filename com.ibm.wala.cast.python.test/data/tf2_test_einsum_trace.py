import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


# Trace einsum (wala/ML#705): in implicit mode the repeated label drops from the
# output, so "ii" contracts the diagonal to a scalar.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
result = tf.einsum("ii", a)
assert isinstance(result, tf.Tensor)
assert result.shape == ()
assert result.dtype == tf.float32
assert float(result) == 5.0
f(result)

# Batch diagonal: the ellipsis and the repeated label compose, so "...ii->...i"
# keeps the batch axis and extracts the diagonal of each square slice.
b = tf.ones((3, 2, 2))
batched = tf.einsum("...ii->...i", b)
assert isinstance(batched, tf.Tensor)
assert batched.shape == (3, 2)
assert batched.dtype == tf.float32
g(batched)
