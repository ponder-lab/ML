import tensorflow as tf


def f(a):
    pass


x = [1, 2, 3, 4, 5]
assert isinstance(x, list)
assert all(isinstance(i, int) for i in x)
assert tf.convert_to_tensor(x).dtype == tf.int32
assert tf.convert_to_tensor(x).shape == (5,)

y = 1
assert isinstance(y, int)
assert tf.convert_to_tensor(y).dtype == tf.int32
assert tf.convert_to_tensor(y).shape == ()

z = tf.add(x, y)
assert isinstance(z, tf.Tensor)
assert z.dtype == tf.int32
assert z.shape == (5,)

f(z)
