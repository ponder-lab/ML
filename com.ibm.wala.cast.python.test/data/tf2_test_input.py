import tensorflow as tf


def f(a):
    pass


arg = tf.keras.Input(shape=[32])
assert arg.shape == (None, 32)
assert arg.dtype == tf.float32

f(arg)
