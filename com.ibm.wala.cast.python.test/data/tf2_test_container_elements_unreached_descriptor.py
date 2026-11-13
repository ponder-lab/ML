# Test `ContainerElements` where a function the call graph does not reach resizes `self.items`, a descriptor whose `__get__` returns a global list
# (wala/ML#1255). The global's list may be stored there, so it reports no length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Shared:
    def __get__(self, obj, owner):
        return G


class Holder:
    items = Shared()

    def build(self):
        self.items.append(1)


G = [x, y]
sink_list(G)
