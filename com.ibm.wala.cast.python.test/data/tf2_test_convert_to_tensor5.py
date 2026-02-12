import tensorflow as tf


def f(a):
    pass


x = tf.convert_to_tensor(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
assert isinstance(x, tf.Tensor)
assert x.dtype == tf.float32
assert x.shape == (2, 2)

f(x)
