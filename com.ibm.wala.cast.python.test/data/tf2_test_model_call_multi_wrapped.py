import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def make_and_call(units, x):
    inputs_layer = tf.keras.layers.Input(shape=(64,))
    outputs_layer = tf.keras.layers.Dense(units)(inputs_layer)
    model = tf.keras.models.Model(inputs_layer, outputs_layer)
    return model(x)


x1 = tf.random.uniform([20, 64])
y1 = make_and_call(5, x1)

x2 = tf.random.uniform([20, 64])
y2 = make_and_call(7, x2)

assert y1.shape == (20, 5)
assert y2.shape == (20, 7)
assert y1.dtype == tf.float32
assert y2.dtype == tf.float32

f(y1)
g(y2)
