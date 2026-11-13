# Test `ContainerElements` where a function the call graph does not reach resizes an attribute of an object whose class defines `__getattr__` (wala/ML#1255).
# The lookup may return any list, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    def __getattr__(self, name):
        return []

    def build(self):
        self.items.append(1)


sink_list([x, y])
