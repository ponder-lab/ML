# wala/ML#817: `tf.keras.Sequential` was unmodeled, so the ubiquitous
# `tf.keras.Sequential([...])` allocated nothing framework-typed: a receiver-typing client could
# not tell a Sequential model from a user object, and the model's call produced no tensor.
import tensorflow as tf


# Deliberate asymmetry with the JUnit expectation: the runtime shape is (3, 2), but the static
# analysis floors the Sequential call's shape at unknown until wala/ML#832's layers walk composes
# the forward chain; the JUnit side pins the definite dtype with the unknown-shape floor.
def consume(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (3, 2)
    assert x.dtype == tf.float32


def consume_transposed(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (2, 3)
    assert x.dtype == tf.float32


model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
out = model(tf.ones((3, 5)))
consume(out)

# A downstream op reading the call's result exercises the producer-delegation route: the
# transpose resolves its input through the allocation in `Sequential/__call__`, the manual half
# of the tandem registration.
transposed = tf.transpose(out)
consume_transposed(transposed)
