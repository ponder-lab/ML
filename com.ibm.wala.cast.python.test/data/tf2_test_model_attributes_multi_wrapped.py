import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def make_model(units):
    inputs_layer = tf.keras.layers.Input(shape=(64,))
    outputs_layer = tf.keras.layers.Dense(units)(inputs_layer)
    return tf.keras.models.Model(inputs_layer, outputs_layer)


model1 = make_model(5)
model2 = make_model(7)

for i in model1.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(64, 5), (5,)]
    f(i)

for j in model2.trainable_weights:
    assert j.dtype == tf.float32
    assert j.shape in [(64, 7), (7,)]
    g(j)
