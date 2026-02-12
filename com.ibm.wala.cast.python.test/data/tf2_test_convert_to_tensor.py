import tensorflow as tf


def f(a):
    pass


y = tf.convert_to_tensor(1)
assert isinstance(y, tf.Tensor)
assert y.dtype == tf.int32
assert y.shape == ()

f(y)
