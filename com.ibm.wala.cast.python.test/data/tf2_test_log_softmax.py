import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.nn.log_softmax(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32

f(y)
