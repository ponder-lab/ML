# wala/ML#817: `tf.keras.Sequential` was unmodeled, so the ubiquitous
# `tf.keras.Sequential([...])` allocated nothing framework-typed: a receiver-typing client could
# not tell a Sequential model from a user object, and the model's call produced no tensor.
import tensorflow as tf


def consume(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
out = model(tf.ones((3, 5)))
consume(out)
