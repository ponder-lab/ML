import tensorflow as tf


def f(a):
    pass


arg = [1.0, 2.0, 3.0, 4.0, 5.0]
assert all(isinstance(i, float) for i in arg)
assert isinstance(arg, list)

x = tf.convert_to_tensor(arg, None, tf.int32)
assert isinstance(x, tf.Tensor)
assert x.dtype == tf.float32
assert x.shape == (5,)

f(x)
