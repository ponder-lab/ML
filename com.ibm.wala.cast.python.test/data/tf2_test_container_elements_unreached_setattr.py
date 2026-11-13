# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, where the program stores an attribute by a computed name (wala/ML#1255).
# The attribute may hold any list, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    def build(self):
        self.items.append(1)


setattr(Holder(), "it" + "ems", [x, y])

sink_list([x, y])
