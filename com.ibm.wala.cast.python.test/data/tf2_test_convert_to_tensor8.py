import tensorflow as tf


def f(a):
    pass


x = tf.convert_to_tensor([1, 2, 3, 4, 5], tf.float32)
assert isinstance(x, tf.Tensor)
assert x.dtype == tf.float32
assert x.shape == (5,)

f(x)
