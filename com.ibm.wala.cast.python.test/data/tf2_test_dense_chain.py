import tensorflow as tf


def consume(tensor):
    pass


inputs = tf.keras.Input(shape=(3,))
assert inputs.shape == (None, 3)
assert inputs.dtype == tf.float32

layer1 = tf.keras.layers.Dense(4)
x = layer1(inputs)
assert x.shape == (None, 4)
assert x.dtype == tf.float32

layer2 = tf.keras.layers.Dense(2)
y = layer2(x)
assert y.shape == (None, 2)
assert y.dtype == tf.float32

consume(y)
