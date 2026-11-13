# Test https://github.com/wala/ML/issues/1249: a model called with a `training` the analysis cannot
# read passes it to its layers, rather than letting them fall back to their default.
import os

import tensorflow as tf

SEEN = []


class Gate(tf.keras.layers.Layer):

    def call(self, x, training=False):
        SEEN.append(training)
        return x


flag = os.environ.get("TRAINING_FLAG") is None
model = tf.keras.Sequential([Gate()])
model(tf.ones([2, 3]), training=flag)
assert set(SEEN) == {True}, SEEN
