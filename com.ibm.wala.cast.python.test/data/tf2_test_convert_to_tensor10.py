import tensorflow as tf


def f(a):
    pass


arg = [1, 2, 3, 4, 5]
assert all(isinstance(i, int) for i in arg)
assert isinstance(arg, list)

x = tf.convert_to_tensor(arg, None, tf.float32)
assert isinstance(x, tf.Tensor)
assert x.dtype == tf.float32
assert x.shape == (5,)

f(x)
