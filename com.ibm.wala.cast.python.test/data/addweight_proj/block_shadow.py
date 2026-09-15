# The shadowing control: the wildcard import also binds `Layer` (the Keras class), and this module
# defines its own `Layer`. The local definition must win as the base class, so `make` resolves and
# `add_weight` does not.
from tf_utils import *


def consume_shadow(m):
    pass


def consume_shadow_weight(w):
    pass


class Layer:
    def make(self):
        return tf.ones((3, 3))


class BlockShadow(Layer):
    def run(self):
        consume_shadow(self.make())
        try:
            consume_shadow_weight(self.add_weight("w", shape=[4, 4], dtype=tf.float32))
        except AttributeError:
            pass
        return self.make()
