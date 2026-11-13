# Test `ContainerElements` where a function the call graph does not reach stores a global list under an attribute and resizes the attribute (wala/ML#1255).
# The global's list is stored there, so it reports no length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    def build(self):
        self.items = G
        self.items.append(1)


G = [x, y]

sink_list(G)
