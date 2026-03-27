import tensorflow as tf


def consume(tensor):
    pass


inputs = tf.keras.Input(shape=(3,))
layer = tf.keras.layers.Dense(4)
x = layer(inputs)
assert x.shape == (None, 4)
assert x.dtype == tf.float32

consume(x)
