# Test for wala/ML#1123: a layer that holds no weights, such as `Dropout`, adds none to a functional
# model's weights, so the dense layers' weights stay known.
import tensorflow as tf


def f(a):
    pass


inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4)(inputs)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(5)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 4), (4,), (4, 5), (5,)]
    f(i)
