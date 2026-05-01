import tensorflow as tf


def add(a, b):
    return a + b


arg = tf.convert_to_tensor(1)
assert arg.shape == ()
assert arg.dtype == tf.int32

c = add(arg, tf.convert_to_tensor(2))
