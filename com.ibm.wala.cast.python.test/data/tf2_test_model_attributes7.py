# From https://github.com/tensorflow/tensorflow/issues/14359#issue-272179775

import tensorflow as tf


def f(a):
    pass


a = tf.keras.layers.Input(shape=(64,))
b = tf.keras.layers.Dense(units=5)(inputs=a)
model = tf.keras.models.Model(inputs=a, outputs=b)

count = 0

for i in model.trainable_weights:
    if count == 0:
        assert i.shape == (64, 5)
    else:
        assert i.shape == (5,)
    count = count + 1

    assert i.dtype == tf.float32

    f(i)
