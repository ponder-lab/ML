# Test for wala/ML#1123: a `Dropout` layer holds no weights and keeps the width of its input, so a
# `Sequential` model with one between its `Dense` layers keeps their weights.
import tensorflow as tf


def f(a):
    pass


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5),
    ]
)
model(tf.ones((2, 3)))

for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 4), (4,), (4, 5), (5,)]
    f(i)
