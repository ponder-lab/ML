import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


a1 = tf.keras.layers.Input(shape=(64,))
b1 = tf.keras.layers.Dense(5)(a1)
model1 = tf.keras.models.Model(a1, b1)

a2 = tf.keras.layers.Input(shape=(32,))
b2 = tf.keras.layers.Dense(7)(a2)
model2 = tf.keras.models.Model(a2, b2)

for i in model1.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(64, 5), (5,)]
    f(i)

for j in model2.trainable_weights:
    assert j.dtype == tf.float32
    assert j.shape in [(32, 7), (7,)]
    g(j)
