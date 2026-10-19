# The functional counterpart of tf2_test_sequential_weights_units_none.py: a Dense layer's `units`
# may, as far as the analysis knows, be `None`. At run time it is 4, but the analysis cannot rule
# `None` out, so the model's weights are tensors of unknown shape rather than a crash.
import tensorflow as tf


def f(a):
    pass


def width(n=None):
    return 4 if n is None else n


inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(width(), activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 4), (4,), (4, 5), (5,)]
    f(i)
