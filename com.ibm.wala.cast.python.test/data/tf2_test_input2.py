import tensorflow as tf


def f(a):
    pass


arg = tf.keras.Input(shape=[None], dtype=tf.string)
assert arg.shape == (None, None)
assert arg.dtype == tf.string

f(arg)
