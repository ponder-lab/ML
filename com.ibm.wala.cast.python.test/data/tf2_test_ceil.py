import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


# Pure passthrough — output shape and dtype both inherit from `x`.
x = tf.constant([1.0, 2.0, 3.0])
assert x.shape == (3,)
assert x.dtype == tf.float32
y = tf.math.ceil(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
g(x)
