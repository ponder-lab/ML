import tensorflow as tf


def add(a, b):
    return a + b

arg1 = tf.random.truncated_normal([2])
assert isinstance(arg1, tf.Tensor)
assert arg1.dtype == tf.float32
assert arg1.shape == (2,)

c = add(arg1, tf.random.truncated_normal([2], 3, 1))
