import tensorflow as tf


def f(a):
    pass


# `tf.nn.leaky_relu` is a pure passthrough — output shape and dtype both
# inherit from `features` (the input).
x = tf.constant([-1.0, 0.0, 2.0])
y = tf.nn.leaky_relu(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
