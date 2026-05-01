import tensorflow as tf


def consume1(tensor):
    pass


def consume2(tensor):
    pass


inputs1 = tf.keras.Input(shape=(3,))
assert inputs1.shape == (None, 3)
assert inputs1.dtype == tf.float32

layer1 = tf.keras.layers.Dense(4)
x1 = layer1(inputs1)
assert x1.shape == (None, 4)
assert x1.dtype == tf.float32
consume1(x1)

inputs2 = tf.keras.Input(shape=(5,))
assert inputs2.shape == (None, 5)
assert inputs2.dtype == tf.float32

layer2 = tf.keras.layers.Dense(2)
x2 = layer2(inputs2)
assert x2.shape == (None, 2)
assert x2.dtype == tf.float32
consume2(x2)
