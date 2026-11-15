# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, `self.items`, under whose name an unrelated Keras layer stores a tensor (wala/ML#1268).
# The layer's tensor can't be the list `Holder.build` appends to, so a list keeps its length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Other(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.items = tf.ones([4, 2])


class Holder:
    def build(self):
        self.items = []
        self.items.append(1)


o = Other()
sink_list([x, y])
