# Test https://github.com/wala/ML/issues/1249: a learning phase the program sets is the `training` a
# layer called outside any layer call gets in place of its default.
import tensorflow as tf

SEEN = []


class Phased(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(training)
        return x


tf.keras.backend.set_learning_phase(1)
Phased()(tf.ones([2, 3]))
assert SEEN == [1], SEEN
