import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


my_tensor = tf.constant([[10, 20, 30], [40, 50, 60]])  # Row 1  # Row 2
assert isinstance(my_tensor, tf.Tensor)
assert my_tensor.dtype == tf.int32
assert my_tensor.shape == (2, 3)

g(my_tensor)

arg = tf.one_hot(my_tensor, 5, None, None, 1)
assert isinstance(arg, tf.Tensor)
assert arg.dtype == tf.float32
assert arg.shape == (2, 5, 3)

f(arg)
