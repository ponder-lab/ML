import tensorflow as tf


def add(a, b):
    return a + b


arg = tf.keras.Input(shape=(32,))
assert arg.shape == (None, 32)
assert arg.dtype == tf.float32

c = add(arg, arg)
