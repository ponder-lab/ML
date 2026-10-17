# Test for wala/ML#1125: a `Dense` layer built with `use_bias=False` has its kernel alone.
import tensorflow as tf


def f(a):
    pass


model = tf.keras.Sequential([tf.keras.layers.Dense(5, use_bias=False)])
model(tf.ones((2, 3)))

for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape == (3, 5)
    f(i)
