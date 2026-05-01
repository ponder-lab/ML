import tensorflow as tf


def f(a):
    pass


value = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
bias = tf.constant([0.1, 0.2, 0.3])
y = tf.nn.bias_add(value=value, bias=bias)
assert isinstance(y, tf.Tensor)
assert y.shape == (2, 3)
assert y.dtype == tf.float32

f(y)
