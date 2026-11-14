# Test `ContainerElements` where a function the call graph does not reach resizes an attribute a class body stores a list under (wala/ML#1255).
# The class's list is stored there, so it reports no length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    items = [x, y]

    def build(self):
        self.items.append(1)


sink_list(Holder.items)
