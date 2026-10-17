# Test for wala/ML#1123: a functional model with a convolution layer has the convolution's kernel and
# bias among its weights, which the analysis does not model, so its weights are tensors of unknown
# shape rather than a dense layer's kernel for the convolution.
import tensorflow as tf


def f(a):
    pass


inputs = tf.keras.Input(shape=(8, 3))
x = tf.keras.layers.Conv1D(4, 3)(inputs)
outputs = tf.keras.layers.Dense(5)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 3, 4), (4,), (4, 5), (5,)]
    f(i)
