# The layer's base class is reached through a wildcard import of a module that imports tensorflow,
# as the vendored transformer's feed-forward module does.
from tf_utils import *


def consume_wild(w):
    pass


class BlockWild(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[4, 4], dtype=tf.float32)
        super(BlockWild, self).build(input_shape)

    def call(self, x):
        consume_wild(self.w)
        return tf.matmul(x, self.w)
