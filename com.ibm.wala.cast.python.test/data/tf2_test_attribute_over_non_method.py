# Test https://github.com/wala/ML/issues/1044.
import tensorflow as tf


def make():
    d = tf.keras.layers.Dense(2)
    # Assigned before any other use, but `trainable` is not a method the model installs.
    d.trainable = False
    return d


layer = make()
y = layer(tf.ones((4, 3)))
assert y.shape == (4, 2)
assert not layer.trainable
